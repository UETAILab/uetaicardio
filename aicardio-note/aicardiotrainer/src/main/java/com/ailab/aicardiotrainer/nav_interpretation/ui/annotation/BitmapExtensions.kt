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

package com.ailab.aicardiotrainer.nav_interpretation.ui.annotation

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.math.abs


/**
 * Converts [Bitmap] to a [Mat].
 */
fun Bitmap.toMat() : Mat {
    val mat = Mat()
    Utils.bitmapToMat(this, mat)
    return mat
}

fun Bitmap.toGray(): Bitmap {
    val mat = Mat()
    mat.toGray(this)
    return mat.toBitmap()
}
//def adjust_gamma(image, gamma=1.0):
//# build a lookup table mapping the pixel values [0, 255] to
//# their adjusted gamma values
//invGamma = 1.0 / gamma
//table = np.array([((i / 255.0) ** invGamma) * 255
//for i in np.arange(0, 256)]).astype("uint8")
//# apply gamma correction using the lookup table
//return cv2.LUT(image, table)

const val MAX_UCHAR = 255.0
const val EPS = 1e-6
fun saturateCastUchar(x: Double): Double {
    return (if (x > MAX_UCHAR) MAX_UCHAR else if (x < 0) 0.0 else x)
}

fun Bitmap.adjustGammaCorrection(gamma: Double = 1.0): Bitmap {
    if (abs(gamma - 1.0) < EPS) return this

    val lookUpTable = Mat(1, 256, CvType.CV_8U)
    val lookUpTableData = ByteArray((lookUpTable.total() * lookUpTable.channels()).toInt())

    for (i in 0 until lookUpTable.cols()) {
//        lookUpTableData[i] = sature(Math.pow(i / 255.0, gamma) * 255.0)
        lookUpTableData[i] = saturateCastUchar(Math.pow(i / 255.0, gamma) * 255.0).toInt().toByte()
    }

    lookUpTable.put(0, 0, lookUpTableData)
    val img = Mat()

    Core.LUT(this.toMat(), lookUpTable, img)

    return img.toBitmap()
}

