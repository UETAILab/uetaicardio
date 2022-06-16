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

package com.ailab.aicardio

import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.Resources
import android.graphics.*
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.ailab.aicardio.repository.*
import com.ailab.aicardio.repository.DicomAnnotation.Companion.EF_BOUNDARY
import com.ailab.aicardio.repository.DicomAnnotation.Companion.EF_POINT
import com.ailab.aicardio.repository.DicomAnnotation.Companion.GLS_BOUNDARY
import com.ailab.aicardio.repository.DicomAnnotation.Companion.GLS_POINT
import com.ailab.aicardio.repository.DicomAnnotation.Companion.IS_EDV
import com.ailab.aicardio.repository.DicomAnnotation.Companion.IS_ESV
import com.ailab.aicardio.repository.DicomAnnotation.Companion.getNewAnnotation
import com.imebra.*
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.*
import java.nio.ByteBuffer
import java.util.logging.Logger
import kotlin.collections.ArrayList
import kotlin.math.abs
import kotlin.math.hypot


const val BASE_URL = "https://newsapi.org/v2/"
//const val API_KEY = BuildConfig.NewsApiKey
const val API_KEY = "40338dc2ef2f41ab9a302a9c750cc6f9"

const val TAG = "AppUtils"
const val TAG_SHORT_CLICKED = "short_clicked"
const val TAG_LONG_CLICKED = "long_clicked"

// https://www.dicomlibrary.com/dicom/dicom-tags/
const val HEX_CODE_NUMBER_FRAME = "(0028,0008)"
const val HEX_CODE_FRAME_DELAY = "(0018,1066)"
const val HEX_CODE_FRAME_TIME = "(0018,1063)"
const val HEX_CODE_PHYSICAL_DELTA_X = "(0018,602C)" // (0008, 0090)
const val HEX_CODE_PHYSICAL_DELTA_Y = "(0018,602E)" // (0008, 0090)
const val HEX_CODE_ROWS = "(0028,0010)"
const val HEX_CODE_COLUMNS = "(0028,0011)"
const val HEX_CODE_SIUID = "(0020,000D)"
const val HEX_CODE_SopIUID = "(0008,0018)"

fun inflate(context: Context, viewId: Int, parent: ViewGroup? = null, attachToRoot: Boolean = false): View {
    return LayoutInflater.from(context).inflate(viewId, parent, attachToRoot)
}

fun Context.toast(message: String, length: Int = Toast.LENGTH_SHORT) {
    Toast.makeText(this, message, length).show()
}

//LCE -> Loading/Content/Error
sealed class LCE<out T> {
    data class Result<T>(val data: T, val error: Boolean = true, val message: String = "error") : LCE<T>()
    data class Error<T>(val message: String) : LCE<T>() {
        constructor(t: Throwable) : this(t.message ?: "")
    }
}


fun getFileJSONFromResources(resources: Resources, idFile: Int) : JSONArray {
    try {
        val inp = InputStreamReader(resources.openRawResource(idFile))

        val str = inp.readText()
        inp.close()
        val result = JSONArray(str)
        Log.w(TAG, "getFileJSONFromResources ${result.length()}")

        return result
//        return@withContext LCE.Result(error = false, message = "Get DATA from Disk Succeed", data = JSONObject(str))

    } catch (e: Exception) {
        Log.w(TAG, "Read getFileJSONFromResources error")
//        return@withContext LCE.Result(error = true, message = "Get DATA from Disk Failed", data = JSONObject())
    }
    return JSONArray()
}

fun getBitmapFromDicom(path: String): Bitmap? {
    val thumbnailFile = getThumbnailFile(path)

    if (thumbnailFile.exists()) {
        Logger.getLogger(TAG).warning("go to thumbnailFile exists")
        return BitmapFactory.decodeFile(thumbnailFile.absolutePath)
    } else {

        val firstFrame = getFrameWithIndexFromDicomFile(path, 0)
        firstFrame?.let {

            val thumnailFolder = File( File(path).parentFile ,".thumbnail")
            if (!thumnailFolder.exists()) {
                thumnailFolder.mkdirs()
            }

            saveThumbnailToStorage(path=thumbnailFile.absolutePath, bitmap=firstFrame)
            return firstFrame
        }
        return null
    }
}


fun saveThumbnailToStorage(path: String, bitmap: Bitmap) {
    try {
        FileOutputStream(path).use({ out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out) // bmp is your Bitmap instance
        })
    } catch (e: IOException) {
        e.printStackTrace()
    }
}

fun getThumbnailFile(path: String): File {
    val thumnailFolder = File( File(path).parentFile ,".thumbnail")

//    if (!thumnailFolder.exists()) {
//        thumnailFolder.mkdirs()
//    }

    val thumbnailBitmap = File(thumnailFolder, File(path).name + ".png")
    return thumbnailBitmap
}

fun getFrameWithIndexFromDicomFile(file_name: String, index_frame: Long): Bitmap? {
    try {
        val loadedDataSet = com.imebra.CodecFactory.load(file_name)

        val image = loadedDataSet.getImageApplyModalityTransform(index_frame)

        // Get the size in pixels
        val width = image.width
        val height = image.height

        // The transforms chain will contain all the transform that we want to
        // apply to the image before displaying it

        val chain = com.imebra.TransformsChain()

        if (com.imebra.ColorTransformsFactory.isMonochrome(image.colorSpace)) {
            // Allocate a VOILUT transform. If the DataSet does not contain any pre-defined
            //  settings then we will find the optimal ones.
            val voilutTransform = VOILUT()

            // Retrieve the VOIs (center/width pairs)
            val vois = loadedDataSet.voIs
            //            com.imebra.
            // Retrieve the LUTs
            val luts = ArrayList<LUT>()
            var scanLUTs: Long = 0
            while (true) {
                try {
                    luts.add(loadedDataSet.getLUT(com.imebra.TagId(0x0028, 0x3010), scanLUTs))
                } catch (e: Exception) {
                    break
                }

                scanLUTs++
            }

            if (!vois.isEmpty) {
                voilutTransform.setCenterWidth(vois.get(0).center, vois.get(0).width)
            } else if (luts.isNotEmpty()) {
                voilutTransform.setLUT(luts[0])
            } else {
                voilutTransform.applyOptimalVOI(image, 0, 0, width, height)
            }

            chain.addTransform(voilutTransform)
        }

        // We create a DrawBitmap that always apply the chain transform before getting the RGB image

        val draw = com.imebra.DrawBitmap(chain)

        //// Ask for the size of the buffer (in bytes)
        val requestedBufferSize = draw.getBitmap(image, drawBitmapType_t.drawBitmapRGBA, 4, ByteArray(0))

        val buffer =
            ByteArray(requestedBufferSize.toInt()) // Ideally you want to reuse this in subsequent calls to getBitmap()
        val byteBuffer = ByteBuffer.wrap(buffer)


        // Now fill the buffer with the image data and create a bitmap from it
        draw.getBitmap(image, drawBitmapType_t.drawBitmapRGBA, 4, buffer)
        val renderBitmap =
            Bitmap.createBitmap(image.width.toInt(), image.height.toInt(), Bitmap.Config.ARGB_8888)
        renderBitmap.copyPixelsFromBuffer(byteBuffer)

        // get the object: renderBitmap
        return renderBitmap

    } catch (e: Exception) {
        //  Block of code to handle errors
        return null

    }

}

fun checkAndRequestPermissions(
    activity: Activity,
    permissions: Array<String>,
    MY_PERMISSIONS_REQUEST_CODE: Int
): Boolean {
    // Here, thisActivity is the current activity
    var granted = true
    permissions.forEach { permission ->
        if (ContextCompat.checkSelfPermission(
                activity,
                permission
            ) != PackageManager.PERMISSION_GRANTED
        )
            granted = false
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

fun getLengthPoint(p1: JSONObject, p2: JSONObject, deltaX: Float, deltaY: Float, nColumn: Float, nRow: Float): Float {
    return try {
        val dx = p1.getDouble("x")-p2.getDouble("x")
        val dy = p1.getDouble("y")-p2.getDouble("y")

        val px = deltaX * nColumn * dx
        val py = deltaY * nRow * dy

        val l = hypot(px, py) *10
        l.toFloat()

    } catch (e: JSONException) {
        Log.w(TAG, "getLengthPoint ${e}")
        0F
    }

}
fun getAreaPath(path: JSONArray, deltaX: Float, deltaY: Float, nColumn: Float, nRow: Float): Float { // mm2

    return try {
        if (path.length() <= 1) return 0F

        val n = path.length()
        var area = 0.0
        val factorial = 50.0F * deltaX * nColumn * deltaY * nRow

        for (i in 0 until n) {
            val ip1 = (i+1)%n
            // get coordinates in mm
//        val p1x = path.getJSONObject(i).getDouble("x") * deltaX * nColumn * 10.0F
//        val p1y = path.getJSONObject(i).getDouble("y") * deltaY * nRow * 10.0F
//        val p2x = path.getJSONObject(ip1).getDouble("x") * deltaX * nColumn * 10.0F
//        val p2y = path.getJSONObject(ip1).getDouble("y") * deltaY * nRow * 10.0F
//        area += (p2x-p1x) * (p1y+p2y) /2.0

            val p1x = path.getJSONObject(i).getDouble("x")
            val p1y = path.getJSONObject(i).getDouble("y")
            val p2x = path.getJSONObject(ip1).getDouble("x")
            val p2y = path.getJSONObject(ip1).getDouble("y")

            area += factorial * (p2x-p1x) * (p1y+p2y)
        }
        area = abs(area)
        area.toFloat()
    } catch (e: JSONException) {
        Log.w(TAG, "getAreaPath ${e}")
        0F

    }


}


fun getMiddelPoint(p0: JSONObject, p6: JSONObject) : JSONObject {
    return try {
        val pMid06 = JSONObject()
        val xMid06 = (p0.getDouble("x") + p6.getDouble("x") ) / 2.0
        val yMid06 = (p0.getDouble("y") + p6.getDouble("y") ) / 2.0
        pMid06.put("x", xMid06.toFloat())
        pMid06.put("y", yMid06.toFloat())
         pMid06
    } catch (e: JSONException) {
        Log.w(TAG, "getMiddelPoint ${e}")
        JSONObject()
    }

}

fun initJSONArray(nFrame: Int, isArray: Boolean = true): JSONArray {
    val o = JSONArray()
    for (i in 0 until nFrame) {
        if (isArray) o.put(JSONArray())
        else o.put(false)
    }
    return o
}


// add to server: version of json
fun converJSONObjectOldVerion(
    oldAnnotation: JSONObject,
    nFrame: Int,
    tags: JSONObject
): Pair<DicomAnnotation, DicomDiagnosis> {

    val o = getNewAnnotation(nFrame)

    var d = DicomDiagnosis()

    // esv, edv
    try {


        // diagnosis
//        Log.w("converJSONObjectOldVerion -- nFrame",     "$nFrame")

        if (oldAnnotation.has("diagnosis")) {
            val diagnosis: JSONObject = oldAnnotation.getJSONObject("diagnosis")
            diagnosis.put(DicomDiagnosis.CHAMBER_IDX, DicomDiagnosis.getChamberIdxFromName(diagnosis.getString(DicomDiagnosis.CHAMBER)))
//            Log.w("converJSONObjectOldVerion", "$diagnosis")
            d = DicomDiagnosis(diagnosis.toString())
        }


        // point
        var ef_points = initJSONArray(nFrame)
        var gls_points = initJSONArray(nFrame)
        var ef_boundary = initJSONArray(nFrame)
        var gls_boundary = initJSONArray(nFrame)


        if (oldAnnotation.has("point")) {
            val points = oldAnnotation.getJSONObject("point")

            if (points.has("frames")){
                val tmp = points.getJSONArray("frames")
                if (tmp.length() == nFrame) ef_points = points.getJSONArray("frames")
            }
            if (points.has("gls") ) {
                val tmp = points.getJSONArray("gls")
                if (tmp.length() == nFrame) gls_points = points.getJSONArray("gls")
            }
        }

        if (oldAnnotation.has("boundary")) {
            val boundary = oldAnnotation.getJSONObject("boundary")
            if (boundary.has("frames")) {
                val tmp = boundary.getJSONArray("frames")

                if (tmp.length() == nFrame) ef_boundary = boundary.getJSONArray("frames")
            }
            if (boundary.has("gls")) {
                val tmp = boundary.getJSONArray("gls")
                if (tmp.length() == nFrame) gls_boundary = boundary.getJSONArray("gls")
            }

        }

        var esv_frames = initJSONArray(nFrame, isArray = false)
        var edv_frames = initJSONArray(nFrame, isArray = false)


        if (oldAnnotation.has("property")) {
            val property = oldAnnotation.getJSONObject("property")
            if (property.has("edv_frames")) {
                val tmp = property.getJSONArray("edv_frames")
                if (tmp.length() == nFrame) edv_frames = tmp
            }
            if (property.has("esv_frames")) {
                val tmp = property.getJSONArray("esv_frames")
                if (tmp.length() == nFrame) esv_frames = tmp
            }

        }


//
//        Log.w("ef_points", "$ef_points")
//        Log.w("gls_points", "$gls_points")
//        Log.w("ef_boundary", "$ef_boundary")
//        Log.w("gls_boundary", "$gls_boundary")
//        Log.w("esv_frames", "$esv_frames")
//        Log.w("edv_frames", "$edv_frames")

        for (i in 0 until nFrame) {

            val fa = DicomAnnotation.getNewFrameAnnotation()

            fa.put(EF_POINT, try { ef_points.getJSONArray(i) } catch (e: JSONException) { JSONArray() } )
            fa.put(EF_BOUNDARY, try { ef_boundary.getJSONArray(i) } catch (e: JSONException) { JSONArray() } )
            fa.put(GLS_POINT, try { gls_points.getJSONArray(i) } catch (e: JSONException) { JSONArray() } )
            fa.put(GLS_BOUNDARY,try { gls_boundary.getJSONArray(i) } catch (e: JSONException) { JSONArray() } )

            fa.put(IS_ESV,  try{esv_frames.getBoolean(i)} catch (e: JSONException) {false} )
            fa.put(IS_EDV, try{edv_frames.getBoolean(i)} catch (e: JSONException) {false} )

            o.put(i, fa)


        }
        o.updateLengthAreaVolumeAllFrame(tags)

    } catch (e: JSONException) {
        Log.w(TAG, "converJSONObjectOldVerion ${e}")

    }
    return Pair(o, d)
}
