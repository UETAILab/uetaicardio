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

import android.graphics.Path
import android.graphics.PathMeasure
import android.util.Log
import com.ailab.aicardiotrainer.*

import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.hypot

class DicomAnnotation: JSONArray {
    // them 1 attribute vao can sua cac ham
    // 1. constructor
    // 2. init new FrameAnnotation getNewFrameAnnotation
    constructor() : super()
    constructor(arr: JSONArray): super() {
        val nFrame = arr.length()
        repeat(nFrame) {
            val o = getNewFrameAnnotation()
            val arr_o = arr.getJSONObject(it)

            o.put(EF_POINT, try { arr_o.getJSONArray(EF_POINT) } catch (e: JSONException) { JSONArray() } )
            o.put(EF_BOUNDARY, try { arr_o.getJSONArray(EF_BOUNDARY) } catch (e: JSONException) { JSONArray() } )
            o.put(GLS_POINT, try { arr_o.getJSONArray(GLS_POINT) } catch (e: JSONException) { JSONArray() } )
            o.put(GLS_BOUNDARY,try { arr_o.getJSONArray(GLS_BOUNDARY) } catch (e: JSONException) { JSONArray() } )

            o.put(IS_ESV,  try{arr_o.getBoolean(IS_ESV)} catch (e: JSONException) {false} )
            o.put(IS_EDV, try{arr_o.getBoolean(IS_EDV)} catch (e: JSONException) {false} )

            o.put(IS_COPY, try{arr_o.getBoolean(IS_COPY)} catch (e: JSONException) {false} )

            o.put(MEASURE_LENGTH, try { arr_o.getJSONArray(MEASURE_LENGTH) } catch (e: JSONException) { JSONArray() } )
            o.put(MEASURE_AREA, try { arr_o.getJSONArray(MEASURE_AREA) } catch (e: JSONException) { JSONArray() } )

            o.put(LENGTH,  try{arr_o.getDouble(LENGTH).toFloat()} catch (e: JSONException) {0F} )
            o.put(AREA,  try{arr_o.getDouble(AREA).toFloat()} catch (e: JSONException) {0F} )
            o.put(VOLUME,  try{arr_o.getDouble(VOLUME).toFloat()} catch (e: JSONException) {0F} )

            o.put(EFObject.KEY_efManual, try{arr_o.getDouble(EFObject.KEY_efManual).toFloat()} catch (e: JSONException) {-1F} )
            o.put(EFObject.KEY_indexFrame, try{arr_o.getInt(EFObject.KEY_indexFrame)} catch (e: JSONException) {-1} )
            o.put(EFObject.KEY_indexESV, try{arr_o.getInt(EFObject.KEY_indexESV)} catch (e: JSONException) {-1} )
            o.put(EFObject.KEY_indexEDV, try{arr_o.getInt(EFObject.KEY_indexEDV)} catch (e: JSONException) {-1} )
            o.put(EFObject.KEY_volumeESV, try{arr_o.getDouble(EFObject.KEY_volumeESV).toFloat()} catch (e: JSONException) {-1F} )
            o.put(EFObject.KEY_volumeEDV, try{arr_o.getDouble(EFObject.KEY_volumeEDV).toFloat()} catch (e: JSONException) {-1F} )
            this.put(it, o)
        }
    }

    class FrameAnnotation : JSONObject()
    class FrameEF : JSONObject()

//    private val annotationRepository: AnnotationRepository = AnnotationRepository.getInstance()

    companion object {
        const val  TAG = "DicomAnnotation"
        const val MOD_EF = "ef"
        const val MOD_GLS = "gls"
        const val MOD_POINT = "_point"
        const val MOD_BOUNDARY = "_boundary"

        const val DISTANT_MODIFY_POINT = 10.0
        const val DISTANT_MODIFY_BOUNDARY = 20.0

        const val EF_POINT = "ef_point"
        const val EF_BOUNDARY = "ef_boundary"
        const val GLS_POINT = "gls_point"
        const val GLS_BOUNDARY = "gls_boundary"

        const val MEASURE_LENGTH = "measure_length"
        const val MEASURE_AREA = "measure_area"

        const val IS_ESV = "is_ESV"
        const val IS_EDV = "is_EDV"

        const val LENGTH = "length"
        const val AREA = "area"
        const val VOLUME = "volume"

        const val IS_COPY = "is_copy"

        const val EPS = 1e-6

        fun getNewFrameAnnotation(): FrameAnnotation {

            val o = FrameAnnotation()
            o.put(EF_POINT, JSONArray())
            o.put(EF_BOUNDARY, JSONArray())
            o.put(GLS_POINT, JSONArray())
            o.put(GLS_BOUNDARY, JSONArray())

            o.put(IS_ESV, false)
            o.put(IS_EDV, false)

            o.put(IS_COPY, false)

            o.put(MEASURE_LENGTH, JSONArray())
            o.put(MEASURE_AREA, JSONArray())

            o.put(LENGTH, 0F)
            o.put(AREA, 0F)
            o.put(VOLUME, 0F)

            o.put(EFObject.KEY_efManual, -1F)
            o.put(EFObject.KEY_indexFrame, -1)
            o.put(EFObject.KEY_indexESV, -1)
            o.put(EFObject.KEY_indexEDV, -1)
            o.put(EFObject.KEY_volumeESV, 0F)
            o.put(EFObject.KEY_volumeEDV, 0F)

            return o
        }

        fun getNewAnnotation(nFrame: Int): DicomAnnotation {
            val o = DicomAnnotation()
            repeat(nFrame) {
                o.put(getNewFrameAnnotation())
            }
            return o
        }
    }


    fun getIsCopy(frameIdx: Int): Boolean {
        return try {
            getJSONObject(frameIdx).getBoolean(IS_COPY)
        } catch (e: JSONException) {
            Log.w(TAG, "getIsCopy ${frameIdx}")
            return false
        }
    }

    fun copyMachineAnnotation(frameIdx: Int, machineAnnotation: DicomAnnotation) {
        // only copy ef_point, ef_boundary, gls_point, gls_boundary
//        val o = machineAnnotation.clone

        try {
            val o = getJSONObject(frameIdx)
            val mo = JSONObject( machineAnnotation.getJSONObject(frameIdx).toString() )
            o.put(EF_POINT, mo.getJSONArray(EF_POINT))
            o.put(EF_BOUNDARY, mo.getJSONArray(EF_BOUNDARY))
            o.put(GLS_POINT, mo.getJSONArray(GLS_POINT))
            o.put(GLS_BOUNDARY, mo.getJSONArray(GLS_BOUNDARY))
        } catch (e: JSONException) {
            Log.w(TAG, "copyMachineAnnotation ${frameIdx}")
        }
    }

    fun copyAllFrameMachineAnnotation(machineAnnotation: DicomAnnotation) {
        // only copy ef_point, ef_boundary, gls_point, gls_boundary
        try {
            repeat(length()) {
                copyMachineAnnotation(it, machineAnnotation)
            }

        } catch (e: JSONException) {
            Log.w(TAG, "copyAllFrameMachineAnnotation")
        }
    }

    fun setIsCopy(frameIdx: Int, is_copy: Boolean) {
        try {
            getJSONObject(frameIdx).put(IS_COPY, is_copy)
        } catch (e: JSONException) {
            Log.w(TAG, "setIsCopy ${frameIdx} ${is_copy}")
        }
    }

    fun setAllFrameIsCopy(is_copy: Boolean = false) {
        try {
            repeat(length()) {
                setIsCopy(it, is_copy)
            }
        } catch (e: JSONException) {
            Log.w(TAG, "setAllFrameIsCopy #nFrame:${length()} COPY: ${is_copy}")

        }
    }



    fun setEFValue(ef: EFObject) {
            try {
                val o = getJSONObject(ef.indexFrame)
                Log.w("setEFValue", "${ef.indexFrame} ${ef.efValue}")
                o.put(EFObject.KEY_efManual, ef.efValue)
                o.put(EFObject.KEY_indexFrame, ef.indexFrame)
                o.put(EFObject.KEY_indexESV, ef.indexESV)
                o.put(EFObject.KEY_indexEDV, ef.indexEDV)
                o.put(EFObject.KEY_volumeESV, ef.volumeESV)
                o.put(EFObject.KEY_volumeEDV, ef.volumeEDV)
            } catch (e: JSONException) {
                Log.w(TAG, "setEFValue ${e}")
            }

    }

    fun chooseModifyPoint(ix: Float, iy: Float, scale: Float, nColumn: Float, nRow: Float, frameIdx: Int, key: String): Int {

        try {
            val points = getPointArray(frameIdx, key)
            val factorC = nColumn * scale
            val factorR = nRow * scale
            repeat(points.length()) {
                val o = points.getJSONObject(it)
                val dx = (o.getDouble("x")-ix) * factorC
                val dy = (o.getDouble("y")-iy) * factorR
                val dist = hypot(dx, dy)
                Log.w(TAG, "$it $dist, $scale $nColumn $nRow")
                if (dist < DISTANT_MODIFY_POINT) {
                    return it
                }
            }
            return -1

        } catch (e: JSONException) {
            Log.w(TAG, "chooseModifyPoint ${key} ${frameIdx} ${e}")
        }
        return -1

    }


    fun addPoint(frameIdx: Int, ix: Float, iy: Float, key: String) {

        val point = JSONObject()
        point.put("x", ix)
        point.put("y", iy)
        try {
            val points = getPointArray(frameIdx, key)
            points.put(point)

        } catch (e: JSONException) {
            Log.w(TAG, "addPoint ${key} ${frameIdx} ${e}")

        }

        // deltaX, deltaY, row, colum

    }

    fun setLastPoint(frameIdx: Int, ix: Float, iy: Float, key: String) {

        val point = JSONObject()
        point.put("x", ix)
        point.put("y", iy)

        try {
            val points = getPointArray(frameIdx, key)
            val n = points.length()

            points.put(n - 1, point)

        } catch (e: JSONException) {
            Log.w(TAG, "setLastPoint ${key} ${frameIdx} ${e}")

        }

    }



    fun getPointArray(frameIdx: Int, key: String): JSONArray {
        try {
            return getJSONObject(frameIdx).getJSONArray(key)
        } catch (e: JSONException) {
            Log.w(TAG, "getPointArray ${key} ${frameIdx} ${e}")
            return JSONArray()
        }

    }
    fun addBoundary(frameIdx: Int, ix: Float, iy: Float, key: String, isNewPath: Boolean) {
        try {
            val point = JSONObject()
            point.put("x", ix)
            point.put("y", iy)

            val boundary = getBoundaryArray(frameIdx = frameIdx, key=key)

            if (isNewPath) boundary.put(JSONArray())

            boundary.getJSONArray(boundary.length() - 1).put(point)

        } catch (e: JSONException) {
            Log.w(TAG, "addBoundary ${key} ${frameIdx} ${e}")

        }

    }

    fun getBoundaryArray(frameIdx: Int, key: String): JSONArray {
        try {
            return getJSONObject(frameIdx).getJSONArray(key)
        } catch (e: JSONException) {
            Log.w(TAG, "getBoundaryArray ${key} ${frameIdx} ${e}")
            return JSONArray()
        }
    }


    fun moveModifyPoint(modifyPointIndex: Int, ix: Float, iy: Float, frameIdx: Int, key: String) {
        try {
            val points = getPointArray(frameIdx, key)
            if (modifyPointIndex >= 0 && modifyPointIndex < points.length()) {
                val o = points.getJSONObject(modifyPointIndex)
                o.put("x", ix)
                o.put("y", iy)
            }
        } catch (e: JSONException) {
            Log.w(TAG, "moveModifyPoint ${key} ${frameIdx} ${e}")
        }

    }

    fun removePoint(frameIdx: Int, key: String) {
        try {
            val points = getPointArray(frameIdx, key)
            val nPoint = points.length()
            if (nPoint > 0)
                points.remove(nPoint - 1)

        } catch (e: JSONException) {
            Log.w(TAG, "removePoint ${key} ${frameIdx} ${e}")

        }


    }

    fun removePath(frameIdx: Int, key: String) {
        try {

            val boundary = getBoundaryArray(frameIdx = frameIdx, key = key)
            val nPath = boundary.length()
            if (nPath > 0) {
                boundary.remove(nPath - 1)
            }

        } catch (e: JSONException) {
            Log.w(TAG, "removePath ${key} ${frameIdx} ${e}")

        }

    }

    fun clearPoints(frameIdx: Int, key: String) {
        try {
            val points = getPointArray(frameIdx, key = key)
            var nPoint = points.length()
            while (nPoint > 0) {
                points.remove(nPoint - 1)
                nPoint -= 1
            }
        } catch (e: JSONException) {
            Log.w(TAG, "clearPoints ${key} ${frameIdx} ${e}")

        }
    }

    fun clearBoundary(frameIdx: Int, key: String) {

        try {
            val boundary = getBoundaryArray(frameIdx = frameIdx, key = key)
            var nPath = boundary.length()
            while (nPath > 0) {
                boundary.remove(nPath - 1)
                nPath -= 1
            }
        } catch (e: JSONException) {
            Log.w(TAG, "clearBoundary ${key} ${frameIdx} ${e}")

        }

    }

    fun setESV(frameIdx: Int, isESV: Boolean) {
        try {
            getJSONObject(frameIdx).put(IS_ESV, isESV)
        } catch (e: JSONException) {
            Log.w(TAG, "setESV ${frameIdx} ${e}")
        }
    }

    fun setEDV(frameIdx: Int, isEDV: Boolean) {
        try {
            getJSONObject(frameIdx).put(IS_EDV, isEDV)
        } catch (e: JSONException) {
            Log.w(TAG, "setEDV ${frameIdx} ${e}")
        }
    }

    fun getIsESVWithFrameIndex(frameIdx: Int): Boolean {
        try {
            return getJSONObject(frameIdx).getBoolean(IS_ESV)
        } catch (e: JSONException) {
            Log.w(TAG, "getIsESVWithFrameIndex ${frameIdx} ${e}")
            return false

        }
    }

    fun getIsEDVWithFrameIndex(frameIdx: Int): Boolean {
        try {
            return getJSONObject(frameIdx).getBoolean(IS_EDV)
        } catch (e: JSONException) {
            Log.w(TAG, "getIsEDVWithFrameIndex ${frameIdx} ${e}")
            return false
        }
    }

    fun getIsAnnotationWithFrameIndex(frameIdx: Int): Boolean {
        try {
            return getPointArray(frameIdx, EF_POINT).length() > 0 && getBoundaryArray(frameIdx, EF_BOUNDARY).length() > 0

        } catch (e: JSONException) {
            Log.w(TAG, "getIsAnnotationWithFrameIndex ${frameIdx} ${e}")
            return false
        }
    }

    fun getNextEsvEdvFrameIndex(nFrame: Int, frameIdx: Int): Int {
        try {
            for(i in 1 until nFrame) {
                val nextFrameIdx = (frameIdx + i) % nFrame
                if (getIsEDVWithFrameIndex(nextFrameIdx) || getIsESVWithFrameIndex(nextFrameIdx))
                    return nextFrameIdx
            }
            return frameIdx

        } catch (e: JSONException) {
            Log.w(TAG, "getNextEsvEdvFrameIndex ${frameIdx} ${e}")
            return frameIdx

        }
    }

    fun getNextAnnoatationFrameIndex(nFrame: Int, frameIdx: Int): Int {

        try {
            for(i in 1 until nFrame) {
                val nextFrameIdx = (frameIdx + i) % nFrame
                if (getIsAnnotationWithFrameIndex(nextFrameIdx))
                    return nextFrameIdx
            }
            return frameIdx

        } catch (e: JSONException) {
            Log.w(TAG, "getNextAnnoatationFrameIndex ${frameIdx} ${e}")
            return frameIdx
        }
    }

    fun chooseModifyBoundary(
        ix: Float,
        iy: Float,
        scale: Float,
        nColumn: Float,
        nRow: Float,
        frameIdx: Int,
        key: String
    ): Pair<Int, Int> {
        try {
            val knots = getKnots(frameIdx, key)
            repeat(knots.length()){i ->

                val knot = knots.getJSONArray(i)
                repeat(knot.length()){ j ->

                    val o = knot.getJSONObject(j)
                    val dx = (o.getDouble("x") - ix) * nColumn * scale
                    val dy = (o.getDouble("y") - iy) * nRow * scale
                    val dist = hypot(dx, dy)
                    Log.w("chooseModifyBoundary Find Index", "$ix $iy ${o.getDouble("x")} ${o.getDouble("y")} $dist")
                    if (dist < DISTANT_MODIFY_BOUNDARY) {
                        return Pair(i, j)
                    }
                }
            }
            return Pair(-1, -1)

        } catch (e: JSONException) {
            Log.w(TAG, "chooseModifyBoundary ${key} ${frameIdx} ${e}")
            return Pair(-1, -1)

        }

    }

    fun moveModifyBoundary(
        ix: Float,
        iy: Float,
        frameIdx: Int,
        key: String,
        bId: Pair<Int, Int> // boundary index
    ) {
        try {
            if (bId.first != -1) {

                val point = JSONObject()
                point.put("x", ix)
                point.put("y", iy)
                val boundarys = getBoundaryArray(frameIdx, key)

                val knots = getKnots(frameIdx, key)

                // push point vao cuoi danh sach cua knot hien tai
                knots.put(bId.first, knots.getJSONArray(bId.first).put(bId.second, point))

                val length = boundarys.getJSONArray(bId.first).length() // lay so diem cua boundary hien tai

                val mBezierSpline =
                    BezierSpline(
                        knots.getJSONArray(bId.first).length()
                    ) // lay Spline cuar knot hien tai

                val newBoundary = mBezierSpline.apply(knots.getJSONArray(bId.first), length)

                // tai cai boundary hien tai no thay the = mot cai knot tu ham Spline
                boundarys.put(bId.first, newBoundary)

            }

        } catch (e: JSONException) {
            Log.w(TAG, "moveModifyBoundary ${key} ${frameIdx} ${e}")

        }


    }

    fun getKnots(frameIdx: Int, key: String, maxKnots: Int = 25): JSONArray {
        try {
            val boundarys = getBoundaryArray(frameIdx, key)

            val knots = JSONArray()

            repeat(boundarys.length()){i ->
                val boundary = boundarys.getJSONArray(i)

                var nKnots =  getNumberOfKnot(boundary) // boundary.length() / 10
                if (nKnots > maxKnots){
                    nKnots = maxKnots
                }
                val knot = JSONArray()
                if (boundary.length() > 0){
                    for (j in 0..nKnots){
                        val index = ((boundary.length()-1).toFloat() * j / nKnots).toInt()
                        val p = boundary.getJSONObject(index)
                        knot.put(p)
                    }
                }
                knots.put(knot)
            }
            return knots
        } catch (e: JSONException) {
            Log.w(TAG, "getKnots ${key} ${frameIdx} ${e}")
            return JSONArray()
        }

    }

    private fun getNumberOfKnot(boundary: JSONArray): Int{
        try {
            val path = Path()
            var p =  boundary.getJSONObject(0)
            path.moveTo(p.getDouble("x").toFloat(), p.getDouble("y").toFloat())
            for (index in 1 until boundary.length()){
                p =  boundary.getJSONObject(index)
                path.lineTo(p.getDouble("x").toFloat(), p.getDouble("y").toFloat())
            }
            val pathMeasure = PathMeasure(path, false)
            val nKnot = pathMeasure.length / 0.05
            return nKnot.toInt()

        } catch (e: JSONException) {
            Log.w(TAG, "getNumberOfKnot ${e}")
            return 0
        }
    }

    fun getMeasureByKey(frameIdx: Int, key: String) : Float{
        try {
            return getJSONObject(frameIdx).getDouble(key).toFloat()
        } catch (e: JSONException) {
            Log.w(TAG, "getMeasureByKey ${key} ${frameIdx} ${e}")
            return 0F
        }
    }

    fun getNFrameAnnotated(): Int {
        var nFrame = 0
        for (i in 0 until this.length()) {
            if (getIsAnnotationWithFrameIndex(i)) nFrame += 1
        }
        return nFrame
    }

    fun changeLength(frameIdx: Int, key: String, tags: JSONObject) {

        if (key != EF_POINT) return

        try {

            val nColumn = if (tags.has(HEX_CODE_COLUMNS)) tags.getString(HEX_CODE_COLUMNS).toFloat() else 636.0F
            val nRow  = if (tags.has(HEX_CODE_ROWS)) tags.getString(HEX_CODE_ROWS).toFloat() else 422.0F
            val deltaX = if (tags.has(HEX_CODE_PHYSICAL_DELTA_X)) tags.getDouble(HEX_CODE_PHYSICAL_DELTA_X).toFloat() else 0.044318250253494909F
            val deltaY = if (tags.has(HEX_CODE_PHYSICAL_DELTA_Y)) tags.getDouble(HEX_CODE_PHYSICAL_DELTA_Y).toFloat() else 0.044318250253494909F

            val o = getJSONObject(frameIdx)
            val points = o.getJSONArray(key)

            if (points.length() == 7) {
                // co du 7 diem thi change length, volume
                val p0 = points.getJSONObject(0)
                val p3 = points.getJSONObject(3)
                val p6 = points.getJSONObject(6)

                val pMid06 = getMiddelPoint(p0, p6)

                val length =  getLengthPoint(p3, pMid06, deltaX, deltaY, nColumn, nRow)
                o.put(LENGTH, length)

                val area = getMeasureByKey(frameIdx, AREA)

                val volume = if (abs(length - 0F) < EPS) 0F else (0.85F * area * area) / length / 1000.0F // mL
//                Log.w(TAG, "L: $length, A: $area V: $volume")

//           val volume = (0.85 * area * area) / length / 1000.0F // mL
                o.put(VOLUME, volume)
//                Log.w("DICOMAnnotation", "${getEFValue(frameIdx)}")


                for (i in 0 until this.length()) {
                    val ef = getEFValue(i)
//                    Log.w(TAG, "frameIdx: ${i} ${ef}")
                    setEFValue(ef)
                }
            }

        } catch (e: JSONException) {
            Log.w(TAG, "changeLength ${key} ${frameIdx} ${e}")

        }


    }



    fun changeArea(frameIdx: Int, key: String, tags: JSONObject) {
        if (key != EF_BOUNDARY) return
        try {
            val nColumn = if (tags.has(HEX_CODE_COLUMNS)) tags.getString(HEX_CODE_COLUMNS).toFloat() else 636.0F
            val nRow  = if (tags.has(HEX_CODE_ROWS)) tags.getString(HEX_CODE_ROWS).toFloat() else 422.0F
            val deltaX = if (tags.has(HEX_CODE_PHYSICAL_DELTA_X)) tags.getDouble(HEX_CODE_PHYSICAL_DELTA_X).toFloat() else 0.044318250253494909F
            val deltaY = if (tags.has(HEX_CODE_PHYSICAL_DELTA_Y)) tags.getDouble(HEX_CODE_PHYSICAL_DELTA_Y).toFloat() else 0.044318250253494909F

            val o = getJSONObject(frameIdx)
            val boundary = o.getJSONArray(key)
            if (boundary.length() >= 1) {
                // chi tinh gia tri voi bounday[0]
                val path = boundary.getJSONArray(0)
                val length = getMeasureByKey(frameIdx, LENGTH)
                val area = getAreaPath(path, deltaX, deltaY, nColumn, nRow)
//                Log.w(TAG, "L: $length, A: $area")
                o.put(AREA, area)

                val volume = if (abs(length - 0F) < EPS) 0F else (0.85F * area * area) / length / 1000.0F // mL
                o.put(VOLUME, volume)

                for (i in 0 until this.length()) {
//                    Log.w(TAG, "frameIdx info: ${i} L = ${getMeasureByKey(i, LENGTH)}, A = ${getMeasureByKey(i, AREA)}, V = ${getMeasureByKey(i, VOLUME)}")
                    val ef = getEFValue(i)
//                    Log.w(TAG, "frameIdx: ${ef}")
                    setEFValue(ef)
                }
            }

        } catch (e: JSONException) {
            Log.w(TAG, "changeArea ${key} ${frameIdx} ${e}")

        }

    }

    fun getPrevFrameIsEDV(frameIdx: Int): Int {
        try {
            for (i in frameIdx downTo 0 step 1) {
                if (this.getIsEDVWithFrameIndex(i)) return i
            }
            return -1
        } catch (e: JSONException) {
            Log.w(TAG, "getPrevFrameIsEDV ${frameIdx} ${e}")
            return -1

        }

    }

    fun getNextFrameIsESV(frameIdx: Int): Int {
        try {
            for (i in frameIdx until this.length()) {
                if (this.getIsESVWithFrameIndex(i)) return i
            }
            return -1
        } catch (e: JSONException) {
            Log.w(TAG, "getNextFrameIsESV ${frameIdx} ${e}")
            return -1
        }
    }

    fun getEFByVolumeAllFrame(): EFObject {
        var minV = 0F
        var maxV = 0F
        var esvIdx = -1
        var edvIdx = -1

        try {

            for (i in 0 until this.length()) {
                val volume = getMeasureByKey(i, VOLUME)
                if ( volume > 0F) {
                    if (volume > maxV) {
                        edvIdx = i
                        maxV = volume
                    }
                    if (esvIdx == -1) {
                        esvIdx = i
                        minV = volume
                    } else if (minV > volume) {
                        minV = volume
                        esvIdx = i
                    }
                }
            }

            var ef = 0F
            if (esvIdx != -1 && edvIdx != -1) {
                ef = getEFFromMinVMaxV(minV, maxV)
            }
//            Log.w("getEFByVolumeAllFrame", "$minV $maxV $esvIdx $edvIdx $ef")
            return EFObject(indexEDV = edvIdx, indexESV = esvIdx, volumeEDV = maxV, volumeESV = minV, efValue = ef)

        } catch (e: JSONException) {
            Log.w(TAG, "getEFByVolumeAllFrame ${e}")
            return EFObject()

        }

    }

    fun updateLengthAreaVolumeAllFrame(tags: JSONObject) {
        try {
            val nColumn = if (tags.has(HEX_CODE_COLUMNS)) tags.getString(HEX_CODE_COLUMNS).toFloat() else 636.0F
            val nRow  = if (tags.has(HEX_CODE_ROWS)) tags.getString(HEX_CODE_ROWS).toFloat() else 422.0F
            val deltaX = if (tags.has(HEX_CODE_PHYSICAL_DELTA_X)) tags.getDouble(HEX_CODE_PHYSICAL_DELTA_X).toFloat() else 0.044318250253494909F
            val deltaY = if (tags.has(HEX_CODE_PHYSICAL_DELTA_Y)) tags.getDouble(HEX_CODE_PHYSICAL_DELTA_Y).toFloat() else 0.044318250253494909F

            for (i in 0 until this.length()) {
                val o = getJSONObject(i)
                val points = getPointArray(i, EF_POINT)
                var length = 0F
                var area = 0F

                if (points.length() >= 7) {
                    val p0 = points.getJSONObject(0)
                    val p3 = points.getJSONObject(3)
                    val p6 = points.getJSONObject(6)

                    val pMid06 = getMiddelPoint(p0, p6)
                    length =  getLengthPoint(p3, pMid06, deltaX, deltaY, nColumn, nRow)
                }

                o.put(LENGTH, length)

                val boundary = getBoundaryArray(i, EF_BOUNDARY)
                if (boundary.length() > 0) {
                    val path = boundary.getJSONArray(0)
                    area = getAreaPath(path, deltaX, deltaY, nColumn, nRow)
                }
                o.put(AREA, area)

                val volume = if (abs(length - 0F) < EPS) 0F else (0.85F * area * area) / length / 1000.0F // mL
                o.put(VOLUME, volume)
            }
            for (i in 0 until this.length()) {
                val ef = getEFValue(i)
                setEFValue(ef)
            }

        } catch (e: JSONException) {
            Log.w(TAG, "updateLengthAreaVolumeAllFrame ${e}")

        }
    }

    fun getEFFromMinVMaxV(minV: Float, maxV: Float): Float {
        return if(abs(maxV - 0F) < EPS ) -1F else (maxV - minV) / maxV
    }
    fun getEFValue(frameIdx: Int): EFObject {

        try {
            val indexEDV = getPrevFrameIsEDV(frameIdx)
            val indexESV = getNextFrameIsESV(frameIdx)

            if (indexEDV != -1 && indexESV != -1) {
                val minV = getMeasureByKey(indexESV, VOLUME)
                val maxV = getMeasureByKey(indexEDV, VOLUME)

//            Log.w("DicomAnnotation---getEFValue", "${EFObject(indexFrame = frameIdx, indexEDV = indexEDV, indexESV = indexESV, volumeEDV = maxV, volumeESV = minV)} $indexEDV $indexESV")

                return EFObject(indexFrame = frameIdx, indexEDV = indexEDV, indexESV = indexESV, volumeEDV = maxV, volumeESV = minV)
            } else {

                val result = getEFByVolumeAllFrame().copy(indexFrame = frameIdx)
//                Log.w("DicomAnnotation---getEFValue", "$result $indexEDV $indexESV")

                if (indexEDV != -1) {
                    val maxV = getMeasureByKey(indexEDV, VOLUME)
                    val minV = result.volumeESV
                    val ef = getEFFromMinVMaxV(minV, maxV)
//                    Log.w("DicomAnnotation---indexEDV != -1", "$result $indexEDV $indexESV $minV $maxV")

                    return result.copy(efValue = ef, volumeEDV = maxV, indexEDV = indexEDV)

                } else if (indexESV != -1) {

                    val minV = getMeasureByKey(indexESV, VOLUME)
                    val maxV = result.volumeEDV
                    val ef = getEFFromMinVMaxV(minV, maxV)

//                    Log.w("DicomAnnotation---indexESV != -1", "$result $indexEDV $indexESV $minV $maxV")

                    return result.copy(efValue = ef, volumeESV = minV, indexESV = indexESV)

                }

                return result
            }

        } catch (e: JSONException) {
            Log.w(TAG, "getEFValue ${frameIdx} ${e}")
            return  EFObject(indexFrame = frameIdx)

        }

    }

    fun getEsvEDVTextDraw(): String {
        var s = "[EDV, ESV]:"
        try {
            for (i in 0 until this.length()) {
                val o =  getJSONObject(i)
                if (o.getBoolean(IS_EDV)) {
                    s = "${s} [${i + 1}"
                }
                if (o.getBoolean(IS_ESV)) {
                    s = "${s} ${i + 1}]"
                }
            }
            return s

        } catch (e: JSONException) {
            Log.w(TAG, "getEsvEDVTextDraw ${e}")
            return s
        }

    }

    fun hasBoundaryAndPoint(frameIdx: Int, key_point: String, key_boundary: String) : Boolean {
        try {
            if (frameIdx < 0 || frameIdx >= length()) return false
            return (getPointArray(frameIdx = frameIdx, key = key_point).length() == 7 &&
                    getBoundaryArray(frameIdx = frameIdx, key = key_boundary).length() > 0)
        } catch (e: JSONException) {
            Log.w(TAG, "hasBoundaryAndPoint ${frameIdx} ${e}")
            return false
        }

    }

    fun getFrameAnnotationObj(frameIdx: Int): JSONObject {
        return try {
            getJSONObject(frameIdx)
        } catch (e: JSONException) {
            Log.w(TAG, "getFrameAnnotationObj ${frameIdx} ${e}")
            getNewFrameAnnotation()
        }
    }

}
