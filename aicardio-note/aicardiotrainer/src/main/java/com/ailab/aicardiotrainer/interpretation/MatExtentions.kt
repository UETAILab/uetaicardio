/*
 * Copyright 2020 UET-AILAB
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ailab.aicardiotrainer.interpretation

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Converts [Mat] to gray scale channel.
 * @param bitmap The [Bitmap] to be converted.
 */
fun Mat.toGray(bitmap: Bitmap) {
    Utils.bitmapToMat(bitmap, this)
    Imgproc.cvtColor(this, this, Imgproc.COLOR_RGB2GRAY)
}



/**
 * Applies the Gaussian Blur algorithm using
 * the provided [Bitmap].
 *
 * @param bitmap The [Bitmap] to be used.
 * @param kSize the Size of the convolution matrix.
 * @param sigmaX Offset on axis X.
 */
inline fun Mat.gaussianBlur(bitmap: Bitmap, kSize: Size = Size(125.toDouble(), 125.toDouble()), sigmaX:Double = 0.toDouble(), block: (Bitmap) -> Unit) {
    Utils.bitmapToMat(bitmap, this)
    Imgproc.GaussianBlur(this, this, kSize, sigmaX)
    return block(this.toBitmap())
}

/**
 * Applies the Canny Edge algorithm in
 * the provided [Bitmap].
 *
 * @param bitmap The [Bitmap] to be used.
 * @param threshold1 The initial threshold.
 * @param threshold2 The final threshold.
 */
inline fun Mat.canny(bitmap: Bitmap, threshold1: Double = 20.toDouble(), threshold2: Double = 255.toDouble(), block: (Bitmap) -> Unit) {
    this.toGray(bitmap)
    Imgproc.Canny(this, this, threshold1, threshold2)
    return block(this.toBitmap())
}

inline fun Mat.canny(bitmap: Bitmap, threshold1: Double = 20.toDouble(), threshold2: Double = 255.toDouble()): Bitmap {
    this.toGray(bitmap)
    Imgproc.Canny(this, this, threshold1, threshold2)
    return this.toBitmap()
}
/**
 * Applies the Canny Edge algorithm in
 * the provided [Bitmap].
 *
 * @param bitmap The [Bitmap] to be used.
 * @param thresh The threshold.
 * @param maxVal Max value for the threshold.
 * @param type The type of the threshold.
 */
fun Mat.threshold(bitmap: Bitmap, thresh: Double = 50.toDouble(), maxVal: Double = 255.toDouble(), type:Int = Imgproc.THRESH_BINARY, block: (Bitmap) -> Unit) {
    this.toGray(bitmap)
    Imgproc.threshold(this, this, thresh, maxVal, type)
    return block(this.toBitmap())
}

/**
 * Applies the Adaptive Threshold algorithm in
 * the provided [Bitmap].
 *
 * @param bitmap The [Bitmap] to be used.
 * @param maxValue The max value for the threshold.
 * @param adaptiveMethod The adaptive method to be used.
 * @param thresholdType The type of the threshold.
 * @param blockSize The block size to be considered.
 * @param c The C parameter value for the algorithm.
 */
fun Mat.adaptiveThreshold(bitmap: Bitmap, maxValue: Double = 255.toDouble(),
                          adaptiveMethod: Int = Imgproc.ADAPTIVE_THRESH_MEAN_C,
                          thresholdType: Int = Imgproc.THRESH_BINARY,
                          blockSize: Int = 11,
                          c: Double = 12.toDouble(), block: (Bitmap) -> Unit) {
    this.toGray(bitmap)
    Imgproc.adaptiveThreshold(this, this, maxValue, adaptiveMethod, thresholdType, blockSize, c)
    return block(this.toBitmap())
}

/**
 * Converts the [Mat] to a [Bitmap].
 *
 * @param config The [Bitmap.Config] to be used.
 */
fun Mat.toBitmap(config: Bitmap.Config = Bitmap.Config.ARGB_8888): Bitmap {
    val bitmap = Bitmap.createBitmap(this.cols(), this.rows(), config)
    Utils.matToBitmap(this, bitmap)
    return bitmap
}

/**
 * Checks if the [Mat] is in gray scale.
 */
fun Mat.inGray() = this.type() == CvType.CV_8U