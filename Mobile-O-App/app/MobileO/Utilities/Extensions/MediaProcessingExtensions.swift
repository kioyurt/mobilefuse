import Accelerate
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM

/// Image processing helpers for the FastVLM vision pipeline (resize, crop, normalize, planar conversion).
enum MediaProcessingExtensions {

    /// Apply user-requested resize processing to a CIImage.
    public static func apply(_ image: CIImage, processing: UserInput.Processing?) -> CIImage {
        var image = image

        if let resize = processing?.resize {
            let scale = MediaProcessing.bestFitScale(image.extent.size, in: resize)
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }

        return image
    }

    /// Check if a rect fits entirely within the given size.
    public static func rectSmallerOrEqual(_ extent: CGRect, size: CGSize) -> Bool {
        return extent.width <= size.width && extent.height <= size.height
    }

    /// Compute a centered crop rect within `extent` clamped to `size`.
    public static func centerCrop(_ extent: CGRect, size: CGSize) -> CGRect {
        let targetWidth = min(extent.width, size.width)
        let targetHeight = min(extent.height, size.height)

        return CGRect(
            x: (extent.maxX - targetWidth) / 2,
            y: (extent.maxY - targetHeight) / 2,
            width: targetWidth, height: targetHeight
        )
    }

    /// Center-crop a CIImage to the given size (no-op if already smaller).
    public static func centerCrop(_ image: CIImage, size: CGSize) -> CIImage {
        let extent = image.extent
        if rectSmallerOrEqual(extent, size: size) {
            return image
        }

        let crop = centerCrop(extent, size: size)
        return
            image
            .cropped(to: crop)
            .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
    }

    /// Scale `size` so its shortest edge matches `shortestEdge`, preserving aspect ratio.
    public static func fitIn(_ size: CGSize, shortestEdge: Int) -> CGSize {
        let floatShortestEdge = CGFloat(shortestEdge)

        let (short, long) =
            size.width <= size.height ? (size.width, size.height) : (size.height, size.width)
        let newShort = floatShortestEdge
        let newLong = floatShortestEdge * long / short

        return size.width <= size.height
            ? CGSize(width: newShort, height: newLong) : CGSize(width: newLong, height: newShort)
    }

    /// Scale `size` so its longest edge matches `longestEdge`, preserving aspect ratio.
    public static func fitIn(_ size: CGSize, longestEdge: Int) -> CGSize {
        let floatLongestEdge = CGFloat(longestEdge)

        var (newShort, newLong) =
            size.width <= size.height ? (size.width, size.height) : (size.height, size.width)

        if newLong > floatLongestEdge {
            newLong = floatLongestEdge
            newShort = floatLongestEdge * newShort / newLong
        }

        return size.width <= size.height
            ? CGSize(width: newShort, height: newLong) : CGSize(width: newLong, height: newShort)
    }

    /// Bicubic resample a CIImage to an exact pixel size.
    public static func resampleBicubic(_ image: CIImage, to size: CGSize) -> CIImage {
        let yScale = size.height / image.extent.height
        let xScale = size.width / image.extent.width

        let filter = CIFilter.bicubicScaleTransform()
        filter.inputImage = image
        filter.scale = Float(yScale)
        filter.aspectRatio = Float(xScale / yScale)
        let scaledImage = filter.outputImage!

        // Crop to ensure exact pixel dimensions after scaling
        return scaledImage.cropped(to: CGRect(origin: .zero, size: size))
    }

    static let context = CIContext()

    /// Convert a CIImage to a planar `[1, 3, H, W]` Float32 MLXArray.
    ///
    /// Uses Accelerate to convert interleaved RGBA → RGB → planar RGB,
    /// which is faster than element-wise readout.
    static public func asPlanarMLXArray(_ image: CIImage, colorSpace: CGColorSpace? = nil)
        -> MLXArray
    {
        let size = image.extent.size
        let w = Int(size.width.rounded())
        let h = Int(size.height.rounded())

        let format = CIFormat.RGBAf
        let componentsPerPixel = 4
        let bytesPerComponent: Int = MemoryLayout<Float32>.size
        let bytesPerPixel = componentsPerPixel * bytesPerComponent
        let bytesPerRow = w * bytesPerPixel

        var data = Data(count: w * h * bytesPerPixel)
        var planarData = Data(count: 3 * w * h * bytesPerComponent)
        data.withUnsafeMutableBytes { ptr in
            context.render(
                image, toBitmap: ptr.baseAddress!, rowBytes: bytesPerRow, bounds: image.extent,
                format: format, colorSpace: colorSpace)
            context.clearCaches()

            let vh = vImagePixelCount(h)
            let vw = vImagePixelCount(w)

            // convert from RGBAf -> RGBf in place
            let rgbBytesPerRow = w * 3 * bytesPerComponent
            var rgbaSrc = vImage_Buffer(
                data: ptr.baseAddress!, height: vh, width: vw, rowBytes: bytesPerRow)
            var rgbDest = vImage_Buffer(
                data: ptr.baseAddress!, height: vh, width: vw, rowBytes: rgbBytesPerRow)

            vImageConvert_RGBAFFFFtoRGBFFF(&rgbaSrc, &rgbDest, vImage_Flags(kvImageNoFlags))

            // and convert to planar data in a second buffer
            planarData.withUnsafeMutableBytes { planarPtr in
                let planeBytesPerRow = w * bytesPerComponent

                var rDest = vImage_Buffer(
                    data: planarPtr.baseAddress!.advanced(by: 0 * planeBytesPerRow * h), height: vh,
                    width: vw, rowBytes: planeBytesPerRow)
                var gDest = vImage_Buffer(
                    data: planarPtr.baseAddress!.advanced(by: 1 * planeBytesPerRow * h), height: vh,
                    width: vw, rowBytes: planeBytesPerRow)
                var bDest = vImage_Buffer(
                    data: planarPtr.baseAddress!.advanced(by: 2 * planeBytesPerRow * h), height: vh,
                    width: vw, rowBytes: planeBytesPerRow)

                vImageConvert_RGBFFFtoPlanarF(
                    &rgbDest, &rDest, &gDest, &bDest, vImage_Flags(kvImageNoFlags))
            }
        }

        return MLXArray(planarData, [1, 3, h, w], type: Float32.self)
    }

}
