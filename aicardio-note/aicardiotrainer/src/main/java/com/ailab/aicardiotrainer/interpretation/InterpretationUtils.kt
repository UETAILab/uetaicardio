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

import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat


data class StudyInstanceUID(

    val studyInstanceUID: String,
    val representationFiles: List<String>,
    val representationBitmap: List<Bitmap>,
    val hashMapBitmap: HashMap<String, Bitmap>

)

data class RenderMP4FrameObject(
    val numFrame: Int,
    val idFrame: Int,
    val inforFrame: String,
    val bitmap: Bitmap,
    val isESV: Boolean,
    val isEDV: Boolean
)


//LCE -> Loading/Content/Error
//sealed class LCE<out T> {
//    data class Result<T>(val data: T, val error: Boolean = true, val message: String = "error") : LCE<T>()
//}

data class LCEResult<T>(val data: T, val error: Boolean = true, val message: String = "error")

fun getResizedBitmap(bm: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
    val width = bm.width
    val height = bm.height
    val scaleWidth = newWidth.toFloat() / width
    val scaleHeight = newHeight.toFloat() / height
    // CREATE A MATRIX FOR THE MANIPULATION
    val matrix = Matrix()
    // RESIZE THE BIT MAP
    matrix.postScale(scaleWidth, scaleHeight)

    // "RECREATE" THE NEW BITMAP
    return Bitmap.createBitmap(bm, 0, 0, width, height, matrix, false)
}

fun checkAndRequestPermissions(
    activity: Activity,
    permissions: Array<String>,
    MY_PERMISSIONS_REQUEST_CODE: Int = InterpretationActivity.MY_PERMISSIONS_REQUEST_CODE
): Boolean {
    // Here, thisActivity is the current activity
    var granted = true
    permissions.forEach { permission ->
        if (ContextCompat.checkSelfPermission(
                activity,
                permission
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            granted = false
//            Log.w(InterpretationActivity.TAG, "checkAndRequestPermissions NOT granted $permission")
        }
        else {
//            Log.w(InterpretationActivity.TAG, "checkAndRequestPermissions granted $permission")
        }
    }

    return if (!granted) {
        // Permission is not granted
        // Should we show an explanation?
        ActivityCompat.requestPermissions(
            activity,
            permissions,
            MY_PERMISSIONS_REQUEST_CODE
        )
        false
    } else {
        true
    }
//        log.warning("Permission granted = $permissionGranted")
}
