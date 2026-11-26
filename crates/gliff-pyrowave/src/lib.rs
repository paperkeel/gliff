//! PyroWave, an intra-only wavelet codec in Vulkan compute, loaded at run
//! time from `libpyrowave-shared.so` (`scripts/build-pyrowave.sh`).
//!
//! gliff uses it only on a direct LAN path, where bandwidth is cheap and the
//! encode and decode times (well under a millisecond at 4K on a desktop GPU)
//! matter more than compression. Without the library, [`Library::load`]
//! fails and gliff stays on H.264.
//!
//! A [`Frame`] on the wire is the encoder's packets, each with a 4-byte
//! little-endian length in front. TCP keeps them in order and whole, so one
//! frame is normally one packet; the packet structure stays so a lossy
//! transport can drop some and still decode.

#[allow(clippy::all)]
mod bindings;

use std::ffi::{c_char, c_void, CStr};
use std::path::PathBuf;
use std::sync::Arc;

use bindings as pw;

/// The library file name, by its ABI major version.
pub const LIBRARY_NAME: &str = "libpyrowave-shared.so.1";

/// The chroma value of grey, as a fraction of full scale, used by both ends
/// of the GPU path (8-bit BT convention: 128/255).
pub const CHROMA_MIDPOINT: f32 = 128.0 / 255.0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("PyroWave is not installed: {0}")]
    NotInstalled(String),
    #[error("PyroWave API {0}.{1} is not supported (need 1.x)")]
    Version(u32, u32),
    #[error("{call} failed: {}", result_name(*.code))]
    Call { call: &'static str, code: i32 },
    #[error("invalid frame: {0}")]
    Frame(&'static str),
    #[error("{0}")]
    Argument(String),
}

pub type Result<T> = std::result::Result<T, Error>;

fn result_name(code: i32) -> String {
    match code {
        pw::PYROWAVE_TIMEOUT => "timeout".into(),
        pw::PYROWAVE_ERROR_GENERIC => "generic error".into(),
        pw::PYROWAVE_ERROR_INVALID_ARGUMENT => "invalid argument".into(),
        pw::PYROWAVE_ERROR_OUT_OF_HOST_MEMORY => "out of host memory".into(),
        pw::PYROWAVE_ERROR_OUT_OF_DEVICE_MEMORY => "out of device memory".into(),
        pw::PYROWAVE_ERROR_NO_VULKAN => "no usable Vulkan device".into(),
        pw::PYROWAVE_ERROR_NOT_IMPLEMENTED => "not implemented".into(),
        pw::PYROWAVE_ERROR_UNSUPPORTED_EXTERNAL_HANDLE => "unsupported external handle".into(),
        pw::PYROWAVE_ERROR_FAILED_EXTERNAL_HANDLE => "external handle import failed".into(),
        other => format!("error {other}"),
    }
}

fn check(call: &'static str, code: pw::pyrowave_result) -> Result<()> {
    if code == pw::PYROWAVE_SUCCESS {
        Ok(())
    } else {
        Err(Error::Call { call, code })
    }
}

/// The loaded library.
pub struct Library {
    api: pw::PyroWave,
    pub path: PathBuf,
    pub version: (u32, u32, u32),
}

impl Library {
    /// Load the library from the first place that has it: `$GLIFF_PYROWAVE_LIB`,
    /// `../lib/gliff/` next to the executable (the package), a
    /// `target/pyrowave/lib/` above it (a development build), then the
    /// default search path.
    pub fn load() -> Result<Arc<Self>> {
        let mut failures = Vec::new();
        for path in candidates() {
            // SAFETY: loading PyroWave runs only its C++ static initializers,
            // and every symbol is checked against the generated signatures.
            match unsafe { pw::PyroWave::new(&path) } {
                Ok(api) => {
                    let (mut major, mut minor, mut patch) = (0, 0, 0);
                    // SAFETY: the out pointers are valid locals.
                    unsafe { api.pyrowave_get_api_version(&mut major, &mut minor, &mut patch) };
                    if major != pw::PYROWAVE_API_VERSION_MAJOR {
                        return Err(Error::Version(major, minor));
                    }
                    tracing::debug!(path = %path.display(), major, minor, patch, "PyroWave loaded");
                    return Ok(Arc::new(Self {
                        api,
                        path,
                        version: (major, minor, patch),
                    }));
                }
                Err(e) => failures.push(format!("{}: {e}", path.display())),
            }
        }
        Err(Error::NotInstalled(failures.join("; ")))
    }
}

fn candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(p) = std::env::var_os("GLIFF_PYROWAVE_LIB") {
        paths.push(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            paths.push(dir.join("../lib/gliff").join(LIBRARY_NAME));
            for up in dir.ancestors() {
                for sub in ["pyrowave/lib", "target/pyrowave/lib"] {
                    let p = up.join(sub).join(LIBRARY_NAME);
                    if p.exists() {
                        paths.push(p);
                    }
                }
            }
        }
    }
    paths.push(PathBuf::from(LIBRARY_NAME));
    paths
}

/// A PyroWave device: its own Vulkan device on the default GPU.
pub struct Device {
    lib: Arc<Library>,
    raw: pw::pyrowave_device,
}

// SAFETY: a pyrowave_device has no thread affinity; gliff uses each one from
// one thread at a time.
unsafe impl Send for Device {}
// SAFETY: as above; shared references only read the handle.
unsafe impl Sync for Device {}

/// The Vulkan objects of a device that PyroWave shares instead of creating
/// its own: it then records into the caller's command buffers and reads and
/// writes the caller's images directly. Handles are raw (dispatchable ones
/// as pointers, the rest as their 64-bit values) so this crate stays
/// independent of any Vulkan binding.
pub struct SharedVulkan {
    /// `vkGetInstanceProcAddr` of the loader that created the instance.
    pub get_instance_proc_addr: *const c_void,
    pub instance: *mut c_void,
    pub physical_device: *mut c_void,
    pub device: *mut c_void,
    /// The create infos the instance and device were made with. They must
    /// stay valid, with everything they point to, as long as the device.
    /// The device's pNext chain must hold a VkPhysicalDeviceFeatures2, and
    /// its queues must include a graphics-capable one.
    pub instance_create_info: *const c_void,
    pub device_create_info: *const c_void,
    /// Queues PyroWave may use for its own uploads: (VkQueue, family, index).
    pub queues: Vec<(*mut c_void, u32, u32)>,
}

impl Device {
    /// Share an existing Vulkan device. Work is then recorded into command
    /// buffers that [`Device::record`] names.
    ///
    /// # Safety
    /// Every handle and create info in `shared` must be valid and outlive
    /// the returned device; the device must enable what PyroWave needs
    /// (8-bit storage, shaderInt16, subgroup size control).
    pub unsafe fn from_vulkan(lib: &Arc<Library>, shared: &SharedVulkan) -> Result<Arc<Self>> {
        let mut queues: Vec<pw::pyrowave_device_create_queue_info> = shared
            .queues
            .iter()
            .map(
                |&(queue, family, index)| pw::pyrowave_device_create_queue_info {
                    queue: queue as pw::VkQueue,
                    familyIndex: family,
                    index,
                },
            )
            .collect();
        let info = pw::pyrowave_device_create_info {
            // SAFETY: the caller passes vkGetInstanceProcAddr, whose
            // signature this is.
            GetInstanceProcAddr: unsafe {
                std::mem::transmute::<*const c_void, pw::PFN_vkGetInstanceProcAddr>(
                    shared.get_instance_proc_addr,
                )
            },
            instance: shared.instance as pw::VkInstance,
            physical_device: shared.physical_device as pw::VkPhysicalDevice,
            device: shared.device as pw::VkDevice,
            instance_create_info: shared.instance_create_info as *const pw::VkInstanceCreateInfo,
            device_create_info: shared.device_create_info as *const pw::VkDeviceCreateInfo,
            queue_info: queues.as_mut_ptr(),
            queue_info_count: queues.len() as u32,
            queue_lock_callback: None,
            queue_unlock_callback: None,
            userdata: std::ptr::null_mut(),
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: the caller vouches for the handles; info and raw are valid.
        check("pyrowave_create_device", unsafe {
            lib.api.pyrowave_create_device(&info, &mut raw)
        })?;
        let device = Arc::new(Self {
            lib: lib.clone(),
            raw,
        });
        // SAFETY: raw is a live device.
        check("pyrowave_device_set_queue_type", unsafe {
            lib.api
                .pyrowave_device_set_queue_type(raw, pw::VK_QUEUE_COMPUTE_BIT)
        })?;
        Ok(device)
    }

    /// Record PyroWave's commands into `cmd` (a raw VkCommandBuffer of a
    /// compute queue) while `f` runs, instead of submitting them itself.
    /// The caller submits `cmd` and waits for it before packetizing an
    /// encode or using a decode.
    pub fn record<T>(&self, cmd: *mut c_void, f: impl FnOnce() -> T) -> T {
        // SAFETY: cmd is in the recording state for the duration of f.
        unsafe {
            self.api()
                .pyrowave_device_set_command_buffer(self.raw, cmd as pw::VkCommandBuffer)
        };
        let out = f();
        // SAFETY: as above; a null handle ends the borrowing.
        unsafe {
            self.api()
                .pyrowave_device_set_command_buffer(self.raw, std::ptr::null_mut())
        };
        out
    }

    pub fn new(lib: &Arc<Library>) -> Result<Arc<Self>> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: raw is a valid out pointer.
        check("pyrowave_create_default_device", unsafe {
            lib.api.pyrowave_create_default_device(&mut raw)
        })?;
        Ok(Arc::new(Self {
            lib: lib.clone(),
            raw,
        }))
    }

    fn api(&self) -> &pw::PyroWave {
        &self.lib.api
    }

    /// GPU timestamps of the passes since the last reset, one line each.
    pub fn performance_stats(&self, reset: bool) -> Vec<String> {
        unsafe extern "C" fn collect(userdata: *mut c_void, msg: *const c_char) {
            // SAFETY: userdata is the Vec below, alive for the call;
            // msg is a NUL-terminated string from PyroWave.
            let (lines, msg) =
                unsafe { (&mut *(userdata as *mut Vec<String>), CStr::from_ptr(msg)) };
            lines.push(msg.to_string_lossy().trim_end().to_string());
        }
        let mut lines: Vec<String> = Vec::new();
        // SAFETY: the callback only runs inside this call.
        unsafe {
            self.api().pyrowave_device_report_performance_stats(
                self.raw,
                Some(collect),
                &mut lines as *mut Vec<String> as *mut c_void,
                reset,
            )
        };
        lines
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: encoders and decoders hold an Arc<Device>, so all of them
        // are gone before the device.
        unsafe { self.api().pyrowave_device_destroy(self.raw) };
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chroma {
    Yuv420,
    Yuv444,
}

impl Chroma {
    fn raw(self) -> pw::pyrowave_chroma_subsampling {
        match self {
            Self::Yuv420 => pw::PYROWAVE_CHROMA_SUBSAMPLING_420,
            Self::Yuv444 => pw::PYROWAVE_CHROMA_SUBSAMPLING_444,
        }
    }

    fn cpu_format(self) -> pw::pyrowave_cpu_buffer_format {
        match self {
            Self::Yuv420 => pw::PYROWAVE_CPU_BUFFER_FORMAT_YUV420P,
            Self::Yuv444 => pw::PYROWAVE_CPU_BUFFER_FORMAT_YUV444P,
        }
    }

    /// The size of plane `i` for a `width` x `height` picture.
    pub fn plane_size(self, i: usize, width: u32, height: u32) -> (u32, u32) {
        match (self, i) {
            (Self::Yuv420, 1 | 2) => (width.div_ceil(2), height.div_ceil(2)),
            _ => (width, height),
        }
    }
}

/// Three tightly packed 8-bit planes (Y, Cb, Cr), full-range BT.709.
pub struct Planes<'a> {
    pub y: &'a [u8],
    pub cb: &'a [u8],
    pub cr: &'a [u8],
}

pub struct PlanesMut<'a> {
    pub y: &'a mut [u8],
    pub cb: &'a mut [u8],
    pub cr: &'a mut [u8],
}

fn cpu_buffer(
    chroma: Chroma,
    width: u32,
    height: u32,
    data: [*mut c_void; 3],
    lens: [usize; 3],
) -> Result<pw::pyrowave_cpu_buffer> {
    let mut row = [0; 3];
    let mut size = [0; 3];
    for (i, len) in lens.iter().enumerate() {
        let (w, h) = chroma.plane_size(i, width, height);
        row[i] = w as usize;
        size[i] = w as usize * h as usize;
        if *len < size[i] {
            return Err(Error::Argument(format!(
                "plane {i} has {len} bytes, needs {}",
                size[i]
            )));
        }
    }
    Ok(pw::pyrowave_cpu_buffer {
        data,
        row_stride_in_bytes: row,
        plane_size_in_bytes: size,
        width: width as i32,
        height: height as i32,
        format: chroma.cpu_format(),
    })
}

/// A view of a caller-owned image for the GPU paths (shared device only).
#[derive(Debug, Clone, Copy)]
pub struct ImageView {
    /// The raw VkImage handle.
    pub image: u64,
    pub width: u32,
    pub height: u32,
    /// VkFormat of the image, and of the view PyroWave reads or writes.
    pub image_format: i32,
    pub view_format: i32,
    /// VkImageLayout the image is in: SHADER_READ_ONLY_OPTIMAL or GENERAL
    /// for an encode input, GENERAL for a decode output.
    pub layout: i32,
}

impl ImageView {
    fn raw(&self) -> pw::pyrowave_image_view {
        pw::pyrowave_image_view {
            image: self.image as usize as pw::VkImage,
            width: self.width,
            height: self.height,
            image_format: self.image_format as pw::VkFormat,
            view_format: self.view_format as pw::VkFormat,
            mip_level: 0,
            layer: 0,
            aspect: pw::VK_IMAGE_ASPECT_COLOR_BIT,
            swizzle: pw::VK_COMPONENT_SWIZZLE_IDENTITY,
            layout: self.layout as pw::VkImageLayout,
        }
    }
}

/// One encoded picture: length-prefixed packets.
#[derive(Debug, Default, Clone)]
pub struct Frame {
    pub data: Vec<u8>,
    pub packets: usize,
}

/// Split a [`Frame`] into its packets.
pub fn frame_packets(mut data: &[u8]) -> impl Iterator<Item = Result<&[u8]>> {
    std::iter::from_fn(move || {
        if data.is_empty() {
            return None;
        }
        if data.len() < 4 {
            data = &[];
            return Some(Err(Error::Frame("truncated packet length")));
        }
        let len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        if data.len() < 4 + len {
            data = &[];
            return Some(Err(Error::Frame("truncated packet")));
        }
        let packet = &data[4..4 + len];
        data = &data[4 + len..];
        Some(Ok(packet))
    })
}

pub struct Encoder {
    device: Arc<Device>,
    raw: pw::pyrowave_encoder,
    pub width: u32,
    pub height: u32,
    pub chroma: Chroma,
    /// The last rate-control budget, which sizes the packetize buffer.
    budget: usize,
    scratch: Vec<u8>,
    packets: Vec<pw::pyrowave_packet>,
}

// SAFETY: see Device; an encoder is used from one thread at a time.
unsafe impl Send for Encoder {}

impl Encoder {
    pub fn new(device: &Arc<Device>, width: u32, height: u32, chroma: Chroma) -> Result<Self> {
        if chroma == Chroma::Yuv420 && (width | height) & 1 != 0 {
            return Err(Error::Argument(format!(
                "4:2:0 needs an even size, got {width}x{height}"
            )));
        }
        let info = pw::pyrowave_encoder_create_info {
            device: device.raw,
            width: width as i32,
            height: height as i32,
            chroma: chroma.raw(),
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: info and raw are valid for the call.
        check("pyrowave_encoder_create", unsafe {
            device.api().pyrowave_encoder_create(&info, &mut raw)
        })?;
        Ok(Self {
            device: device.clone(),
            raw,
            width,
            height,
            chroma,
            budget: 0,
            scratch: Vec::new(),
            packets: Vec::new(),
        })
    }

    /// Encode planes from memory within `max_bytes` (exact rate control:
    /// the bitstream never exceeds it). Blocks until the GPU is done.
    pub fn encode_planes(&mut self, planes: &Planes<'_>, max_bytes: usize) -> Result<()> {
        let buffer = cpu_buffer(
            self.chroma,
            self.width,
            self.height,
            [
                planes.y.as_ptr() as *mut c_void,
                planes.cb.as_ptr() as *mut c_void,
                planes.cr.as_ptr() as *mut c_void,
            ],
            [planes.y.len(), planes.cb.len(), planes.cr.len()],
        )?;
        let rate = pw::pyrowave_rate_control {
            maximum_bitstream_size: max_bytes,
        };
        self.budget = max_bytes;
        // SAFETY: the encoder only reads the planes (the const cast is for
        // the shared CPU buffer struct), and they outlive the call.
        check("pyrowave_encoder_encode_cpu", unsafe {
            self.device
                .api()
                .pyrowave_encoder_encode_cpu(self.raw, &buffer, &rate)
        })
    }

    /// Record an encode of an RGB(A) UNORM image, scaled to the encoder's
    /// size and converted to full-range BT.709 YCbCr, into the command
    /// buffer set with [`Device::record`]. Submit and wait for it before
    /// [`Encoder::packetize`].
    pub fn encode_gpu(&mut self, input: &ImageView, max_bytes: usize) -> Result<()> {
        let info = pw::pyrowave_scaled_encode_info {
            view: input.raw(),
            input_color_space: pw::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            output_color_space: pw::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            // 16-bit planes: no dither noise on text.
            intermediate_plane_format: pw::VK_FORMAT_R16_UNORM,
            ycbcr_chroma_midpoint: CHROMA_MIDPOINT,
            force_linear_filtering: false,
            skip_dither: true,
            crop_rect: std::ptr::null(),
            ycbcr_range: pw::VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
            ycbcr_range_bit_depth: 0,
        };
        let rate = pw::pyrowave_rate_control {
            maximum_bitstream_size: max_bytes,
        };
        self.budget = max_bytes;
        // SAFETY: the view names a live image in the stated layout; sync is
        // left to the caller's command buffer (both sync operations null).
        check("pyrowave_encoder_encode_gpu_scaled", unsafe {
            self.device.api().pyrowave_encoder_encode_gpu_scaled(
                self.raw,
                std::ptr::null(),
                std::ptr::null(),
                &info,
                &rate,
            )
        })
    }

    /// The last encode as a [`Frame`] of packets of at most `boundary` bytes.
    pub fn packetize(&mut self, boundary: usize) -> Result<Frame> {
        let api = self.device.api();
        let mut count = 0;
        // SAFETY: count is a valid out pointer; an encode came before.
        check("pyrowave_encoder_compute_num_packets", unsafe {
            api.pyrowave_encoder_compute_num_packets(self.raw, boundary, &mut count)
        })?;
        self.packets.clear();
        self.packets
            .resize(count, pw::pyrowave_packet { offset: 0, size: 0 });
        // The bitstream fits the budget plus its metadata and headers.
        self.scratch.resize(self.budget + 64 * 1024 + count * 64, 0);
        let mut written = 0;
        // SAFETY: packets holds `count` entries and scratch its stated size.
        check("pyrowave_encoder_packetize", unsafe {
            api.pyrowave_encoder_packetize(
                self.raw,
                self.packets.as_mut_ptr(),
                boundary,
                &mut written,
                self.scratch.as_mut_ptr() as *mut c_void,
                self.scratch.len(),
            )
        })?;
        let mut frame = Frame {
            data: Vec::with_capacity(self.scratch.len() + written * 4),
            packets: written,
        };
        for p in &self.packets[..written] {
            let bytes = self
                .scratch
                .get(p.offset..p.offset + p.size)
                .ok_or(Error::Frame("packet outside the bitstream"))?;
            frame.data.extend_from_slice(&(p.size as u32).to_le_bytes());
            frame.data.extend_from_slice(bytes);
        }
        Ok(frame)
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // SAFETY: raw came from pyrowave_encoder_create; the device lives on.
        unsafe { self.device.api().pyrowave_encoder_destroy(self.raw) };
    }
}

pub struct Decoder {
    device: Arc<Device>,
    raw: pw::pyrowave_decoder,
    pub width: u32,
    pub height: u32,
    pub chroma: Chroma,
}

// SAFETY: see Device; a decoder is used from one thread at a time.
unsafe impl Send for Decoder {}

impl Decoder {
    pub fn new(device: &Arc<Device>, width: u32, height: u32, chroma: Chroma) -> Result<Self> {
        let info = pw::pyrowave_decoder_create_info {
            device: device.raw,
            width: width as i32,
            height: height as i32,
            chroma: chroma.raw(),
            fragment_path: false,
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: info and raw are valid for the call.
        check("pyrowave_decoder_create", unsafe {
            device.api().pyrowave_decoder_create(&info, &mut raw)
        })?;
        Ok(Self {
            device: device.clone(),
            raw,
            width,
            height,
            chroma,
        })
    }

    /// Queue every packet of `frame`. Returns whether the picture is
    /// complete. A whole frame that the decoder does not take is from a
    /// restarted encoder (its sequence counter went back), so the decoder
    /// forgets the old sequence and takes it.
    pub fn push_frame(&mut self, frame: &[u8]) -> Result<bool> {
        if self.push_packets(frame)? {
            return Ok(true);
        }
        // SAFETY: raw is a live decoder.
        unsafe { self.device.api().pyrowave_decoder_clear(self.raw) };
        self.push_packets(frame)
    }

    fn push_packets(&mut self, frame: &[u8]) -> Result<bool> {
        let api = self.device.api();
        for packet in frame_packets(frame) {
            let packet = packet?;
            // SAFETY: the decoder copies the packet before returning.
            check("pyrowave_decoder_push_packet", unsafe {
                api.pyrowave_decoder_push_packet(
                    self.raw,
                    packet.as_ptr() as *const c_void,
                    packet.len(),
                )
            })?;
        }
        // SAFETY: raw is a live decoder.
        Ok(unsafe { api.pyrowave_decoder_decode_is_ready(self.raw, false) })
    }

    /// Record a decode of the queued picture into three single-channel
    /// images (Y, Cb, Cr, STORAGE usage, GENERAL layout) into the command
    /// buffer set with [`Device::record`].
    pub fn decode_gpu(&mut self, planes: &[ImageView; 3]) -> Result<()> {
        let buffers = pw::pyrowave_gpu_buffers {
            planes: [planes[0].raw(), planes[1].raw(), planes[2].raw()],
        };
        // SAFETY: the views name live images in GENERAL layout; sync is left
        // to the caller's command buffer.
        check("pyrowave_decoder_decode_gpu_buffer", unsafe {
            self.device.api().pyrowave_decoder_decode_gpu_buffer(
                self.raw,
                std::ptr::null(),
                std::ptr::null(),
                &buffers,
            )
        })
    }

    /// Decode the queued picture into memory. Blocks until the GPU is done.
    pub fn decode_planes(&mut self, planes: &mut PlanesMut<'_>) -> Result<()> {
        let buffer = cpu_buffer(
            self.chroma,
            self.width,
            self.height,
            [
                planes.y.as_mut_ptr() as *mut c_void,
                planes.cb.as_mut_ptr() as *mut c_void,
                planes.cr.as_mut_ptr() as *mut c_void,
            ],
            [planes.y.len(), planes.cb.len(), planes.cr.len()],
        )?;
        // SAFETY: the planes are writable, sized by cpu_buffer, and outlive
        // the synchronous call.
        check("pyrowave_decoder_decode_cpu_buffer_synchronous", unsafe {
            self.device
                .api()
                .pyrowave_decoder_decode_cpu_buffer_synchronous(self.raw, &buffer)
        })
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: raw came from pyrowave_decoder_create; the device lives on.
        unsafe { self.device.api().pyrowave_decoder_destroy(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_split_into_their_packets() {
        let mut data = Vec::new();
        for p in [&b"abc"[..], &b""[..], &b"de"[..]] {
            data.extend_from_slice(&(p.len() as u32).to_le_bytes());
            data.extend_from_slice(p);
        }
        let packets: Vec<_> = frame_packets(&data).collect::<Result<_>>().unwrap();
        assert_eq!(packets, [&b"abc"[..], &b""[..], &b"de"[..]]);
        assert!(frame_packets(&data[..data.len() - 1]).any(|p| p.is_err()));
        assert!(frame_packets(&[1, 0]).any(|p| p.is_err()));
    }

    #[test]
    fn chroma_plane_sizes() {
        assert_eq!(Chroma::Yuv444.plane_size(1, 3840, 2160), (3840, 2160));
        assert_eq!(Chroma::Yuv420.plane_size(2, 3840, 2160), (1920, 1080));
        assert_eq!(Chroma::Yuv420.plane_size(0, 3840, 2160), (3840, 2160));
    }

    /// Round-trips a picture when the library and a GPU are present
    /// (`scripts/build-pyrowave.sh`); skips otherwise.
    #[test]
    fn round_trip_when_available() {
        let Ok(lib) = Library::load() else {
            eprintln!("PyroWave not built; skipping");
            return;
        };
        let Ok(device) = Device::new(&lib) else {
            eprintln!("no PyroWave device; skipping");
            return;
        };
        let (w, h) = (256, 128);
        let n = (w * h) as usize;
        // Dark text-like content: the mean sits far from mid-grey, so a
        // lost low band (the picture's DC) fails the check.
        let y: Vec<u8> = (0..n)
            .map(|i| {
                if (i % 7) < 2 && (i / w as usize) % 9 < 6 {
                    220
                } else {
                    30
                }
            })
            .collect();
        let (cb, cr) = (vec![100u8; n], vec![160u8; n]);
        let mut enc = Encoder::new(&device, w, h, Chroma::Yuv444).unwrap();
        enc.encode_planes(
            &Planes {
                y: &y,
                cb: &cb,
                cr: &cr,
            },
            n * 3,
        )
        .unwrap();
        let frame = enc.packetize(64 * 1024).unwrap();
        assert!(frame.packets >= 1 && frame.data.len() <= n * 3 + 64 * 1024);
        let mut dec = Decoder::new(&device, w, h, Chroma::Yuv444).unwrap();
        assert!(dec.push_frame(&frame.data).unwrap());
        let (mut y2, mut cb2, mut cr2) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
        dec.decode_planes(&mut PlanesMut {
            y: &mut y2,
            cb: &mut cb2,
            cr: &mut cr2,
        })
        .unwrap();
        let err = y
            .iter()
            .zip(&y2)
            .map(|(a, b)| a.abs_diff(*b) as u32)
            .max()
            .unwrap();
        assert!(err <= 8, "max luma error {err}");
        let mean = |v: &[u8]| v.iter().map(|&x| x as u64).sum::<u64>() / v.len() as u64;
        assert!(mean(&y).abs_diff(mean(&y2)) <= 1);
        assert!(cb2.iter().all(|v| v.abs_diff(100) <= 2));
    }
}
