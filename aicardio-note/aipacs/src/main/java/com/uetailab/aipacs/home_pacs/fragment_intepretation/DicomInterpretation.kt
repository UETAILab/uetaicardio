/*
 * Copyright 2021 UET-AILAB
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

package com.uetailab.aipacs.home_pacs.fragment_intepretation

import android.graphics.Path
import android.graphics.PathMeasure
import android.util.Log
import com.uetailab.aipacs.home_pacs.fragment_intepretation.FrameAnnotation.Companion.AREA
import com.uetailab.aipacs.home_pacs.fragment_intepretation.FrameAnnotation.Companion.EF_BOUNDARY
import com.uetailab.aipacs.home_pacs.fragment_intepretation.FrameAnnotation.Companion.EF_POINT
import com.uetailab.aipacs.home_pacs.fragment_intepretation.FrameAnnotation.Companion.IS_EDV
import com.uetailab.aipacs.home_pacs.fragment_intepretation.FrameAnnotation.Companion.IS_ESV
import com.uetailab.aipacs.home_pacs.fragment_intepretation.FrameAnnotation.Companion.LENGTH
import com.uetailab.aipacs.home_pacs.fragment_intepretation.FrameAnnotation.Companion.VOLUME
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewDrawTouchEventMVI.Companion.getAreaPath
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewDrawTouchEventMVI.Companion.getLengthPoint
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewDrawTouchEventMVI.Companion.getMiddelPoint
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewState.Companion.HEX_CODE_COLUMNS
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewState.Companion.HEX_CODE_PHYSICAL_DELTA_X
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewState.Companion.HEX_CODE_PHYSICAL_DELTA_Y
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewState.Companion.HEX_CODE_ROWS
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.hypot

class DicomInterpretation: JSONObject {

    constructor(): super() {
        put(KEY_DIAGNOSIS, DicomDiagnosis())
        put(KEY_ANNOTATION, JSONArray())
    }


    constructor(o: JSONObject) {
        // check number of frame != size of o["KEY_ANNOTATION"]
        put(KEY_DIAGNOSIS, try { DicomDiagnosis(o.getJSONObject(KEY_DIAGNOSIS)) } catch (e: JSONException) { DicomDiagnosis() } )
        try {
            val newAnnotation = JSONArray()
            val annotation = o.getJSONArray(KEY_ANNOTATION)
            repeat(annotation.length()) {
                newAnnotation.put(FrameAnnotation(annotation.getJSONObject(it)))
            }
            put(KEY_ANNOTATION, newAnnotation)
        } catch (e: Exception) {
            put(KEY_ANNOTATION, JSONArray())
        }
    }

    constructor(numFrame: Int) {
        Log.w(TAG, "init file numFrame: ${numFrame}")
        put(KEY_DIAGNOSIS, DicomDiagnosis())
        val dicomAnnotation = JSONArray()
        repeat(numFrame) {
            Log.w(TAG, "constructor: ${numFrame} ${it}")
            dicomAnnotation.put(FrameAnnotation())
        }
        put(KEY_ANNOTATION, dicomAnnotation)
    }

    companion object {
        const val TAG = "DicomInterpretation"
        const val KEY_ANNOTATION = "annotation"
        const val KEY_DIAGNOSIS = "diagnosis"
        const val DISTANT_MODIFY_POINT = 10.0
        const val DISTANT_MODIFY_BOUNDARY = 20.0
        const val EPS = 1e-6


    }


//    val dicomAnnotation: JSONArray = get(KEY_ANNOTATION) as JSONArray
//    val dicomDiagnosis: DicomDiagnosis =
//        if (has(KEY_DIAGNOSIS)) get(KEY_DIAGNOSIS) as DicomDiagnosis else DicomDiagnosis()

    fun hasBoundaryAndPoint(frameIdx: Int, key_point: String, key_boundary: String) : Boolean {
        try {
            val o = getDataFrame(frameIdx)

            if (frameIdx < 0 || frameIdx >= o.length()) return false
            return (o.getJSONArray(key_point).length() == 7 &&
                    o.getJSONArray(key_boundary).length()  > 0)

        } catch (e: JSONException) {
            Log.w(TAG, "hasBoundaryAndPoint ${frameIdx} ${e}")
            return false
        }

    }

    fun getDicomDiagnosisTest(): DicomDiagnosis {
        return get(KEY_DIAGNOSIS) as DicomDiagnosis
    }
    fun getDataFrame(frameIdx: Int): JSONObject {
        try {
            val dicomAnnotation: JSONArray = get(KEY_ANNOTATION) as JSONArray
            return dicomAnnotation.getJSONObject(frameIdx)
        } catch (e: JSONException) {
            return JSONObject()
        }
    }

    fun getFramePointArrayWithKey(frameIdx: Int, key: String): JSONArray {

        try {
            val dicomAnnotation: JSONArray = get(KEY_ANNOTATION) as JSONArray
            return dicomAnnotation.getJSONObject(frameIdx).getJSONArray(key)
        } catch (e: JSONException) {
            Log.w(TAG, "getFramePointArrayWithKey ${key} ${frameIdx} ${e}")
            return JSONArray()
        }
    }

    fun  addPoint(frameIdx: Int, ix: Float, iy: Float, key: String) {
        val point = JSONObject()
        point.put("x", ix)
        point.put("y", iy)
        try {
            val points = getFramePointArrayWithKey(frameIdx, key)
            points.put(point)
        } catch (e: JSONException) {
            Log.w(TAG, "addPoint ${key} ${frameIdx} ${e}")
        }
    }

    fun setLastPoint(frameIdx: Int, ix: Float, iy: Float, key: String) {

        val point = JSONObject()
        point.put("x", ix)
        point.put("y", iy)

        try {

            val points = getFramePointArrayWithKey(frameIdx, key)
            val n = points.length()
            points.put(n - 1, point)

        } catch (e: JSONException) {
            Log.w(TAG, "setLastPoint ${key} ${frameIdx} ${e}")

        }

    }


    fun addBoundary(frameIdx: Int, ix: Float, iy: Float, key: String, isNewPath: Boolean) {
        try {
            val point = JSONObject()
            point.put("x", ix)
            point.put("y", iy)

            val boundary = getFramePointArrayWithKey(frameIdx = frameIdx, key=key)
            if (isNewPath) boundary.put(JSONArray())

            boundary.getJSONArray(boundary.length() - 1).put(point)

        } catch (e: JSONException) {
            Log.w(TAG, "addBoundary ${key} ${frameIdx} ${e}")

        }
    }

    fun chooseModifyPoint(ix: Float, iy: Float, scale: Float, nColumn: Float, nRow: Float, frameIdx: Int, key: String): Int {

        try {
            val points = getFramePointArrayWithKey(frameIdx, key)
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

    fun moveModifyPoint(modifyPointIndex: Int, ix: Float, iy: Float, frameIdx: Int, key: String) {
        try {
            val points = getFramePointArrayWithKey(frameIdx, key)
            if (modifyPointIndex >= 0 && modifyPointIndex < points.length()) {
                val o = points.getJSONObject(modifyPointIndex)
                o.put("x", ix)
                o.put("y", iy)
            }
        } catch (e: JSONException) {
            Log.w(TAG, "moveModifyPoint ${key} ${frameIdx} ${e}")
        }

    }

    fun chooseModifyBoundary( ix: Float, iy: Float, scale: Float, nColumn: Float, nRow: Float, frameIdx: Int, key: String ): Pair<Int, Int> {
        try {
            val knots = getKnots(frameIdx, key)

//            if (knots.length() <= 2) return Pair(-1, -1)

            repeat(knots.length()){i ->

                val knot = knots.getJSONArray(i)

                if (knot.length() <= 2) return Pair(-1, -1)

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

    fun moveModifyBoundary( ix: Float, iy: Float, frameIdx: Int, key: String, bId: Pair<Int, Int>) {

        try {
            if (bId.first != -1) {

                val point = JSONObject()
                point.put("x", ix)
                point.put("y", iy)
                val boundarys = getFramePointArrayWithKey(frameIdx, key)

                val knots = getKnots(frameIdx, key)

                // push point vao cuoi danh sach cua knot hien tai
                knots.put(bId.first, knots.getJSONArray(bId.first).put(bId.second, point))

                val length = boundarys.getJSONArray(bId.first).length() // lay so diem cua boundary hien tai

                val mBezierSpline = BezierSpline(knots.getJSONArray(bId.first).length()) // lay Spline cuar knot hien tai

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
            val boundarys = getFramePointArrayWithKey(frameIdx, key)

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

    fun setESV(frameIdx: Int, isESV: Boolean) {
        try {
            getJSONArray(KEY_ANNOTATION).getJSONObject(frameIdx).put(IS_ESV, isESV)
        } catch (e: JSONException) {
            Log.w(TAG, "setESV ${frameIdx} ${e}")
        }
    }

    fun setEDV(frameIdx: Int, isEDV: Boolean) {
        try {
            getJSONArray(KEY_ANNOTATION).getJSONObject(frameIdx).put(IS_EDV, isEDV)
        } catch (e: JSONException) {
            Log.w(TAG, "setEDV ${frameIdx} ${e}")
        }
    }

    fun getIsESVWithFrameIndex(frameIdx: Int): Boolean {
        try {
            return getJSONArray(KEY_ANNOTATION).getJSONObject(frameIdx).getBoolean(IS_ESV)
        } catch (e: JSONException) {
            Log.w(TAG, "getIsESVWithFrameIndex ${frameIdx} ${e}")
            return false
        }
    }

    fun getIsEDVWithFrameIndex(frameIdx: Int): Boolean {
        try {
            return getJSONArray(KEY_ANNOTATION).getJSONObject(frameIdx).getBoolean(IS_EDV)
        } catch (e: JSONException) {
            Log.w(TAG, "getIsEDVWithFrameIndex ${frameIdx} ${e}")
            return false
        }
    }


    fun removePointBoundary(frameIdx: Int, key: String) {
        try {
            val points = getJSONArray(KEY_ANNOTATION).getJSONObject(frameIdx).getJSONArray(key)
            val nPoint = points.length()
            if (nPoint > 0)
                points.remove(nPoint - 1)

        } catch (e: JSONException) {
            Log.w(TAG, "removePoint ${key} ${frameIdx} ${e}")

        }


    }

    fun clearPointsBoundary(frameIdx: Int, key: String) {
        try {
            val points = getJSONArray(KEY_ANNOTATION).getJSONObject(frameIdx).getJSONArray(key)
            var nPoint = points.length()
            while (nPoint > 0) {
                points.remove(nPoint - 1)
                nPoint -= 1
            }
        } catch (e: JSONException) {
            Log.w(TAG, "clearPoints ${key} ${frameIdx} ${e}")

        }
    }

    fun removeClearPointsBoundary(frameIdx: Int, key: String, isClear: Boolean) {
        try {
            val points = getJSONArray(KEY_ANNOTATION).getJSONObject(frameIdx).getJSONArray(key)
            var nPoint = points.length()
            if (isClear) {
                while (nPoint > 0) {
                    points.remove(nPoint - 1)
                    nPoint -= 1
                }
            } else {
                if (nPoint > 0)
                    points.remove(nPoint - 1)
            }
        } catch (e: JSONException) {
            Log.w(TAG, "clearPoints ${key} ${frameIdx} ${e}")

        }
    }

    fun setDicomDiagnosis(dicomDiagnosis: DicomDiagnosis) {
        Log.w(TAG, "setDicomDiagnosis: ${dicomDiagnosis}")
        put(KEY_DIAGNOSIS, dicomDiagnosis)
        Log.w(TAG, "setDicomDiagnosis ${this}")

    }

    fun updateLengthAreaVolumeAllFrame(tags: JSONObject) {
        try {
            Log.w(TAG, "updateLengthAreaVolumeAllFrame: ${tags}")
            val nColumn = if (tags.has(HEX_CODE_COLUMNS)) tags.getString(HEX_CODE_COLUMNS).toFloat() else 636.0F
            val nRow  = if (tags.has(HEX_CODE_ROWS)) tags.getString(HEX_CODE_ROWS).toFloat() else 422.0F
            val deltaX = if (tags.has(HEX_CODE_PHYSICAL_DELTA_X)) tags.getDouble(HEX_CODE_PHYSICAL_DELTA_X).toFloat() else 0.044318250253494909F
            val deltaY = if (tags.has(HEX_CODE_PHYSICAL_DELTA_Y)) tags.getDouble(HEX_CODE_PHYSICAL_DELTA_Y).toFloat() else 0.044318250253494909F

            val annotation = getJSONArray(KEY_ANNOTATION)
            for (i in 0 until annotation.length()) {
                val o = annotation.getJSONObject(i)

                val points = o.getJSONArray(EF_POINT)
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

                val boundary = o.getJSONArray(EF_BOUNDARY)

                if (boundary.length() > 0) {
                    val path = boundary.getJSONArray(0)
                    area = getAreaPath(path, deltaX, deltaY, nColumn, nRow)
                }

                o.put(AREA, area)

                val volume = if (abs(length - 0F) < EPS) 0F else (0.85F * area * area) / length / 1000.0F // mL
                o.put(VOLUME, volume)
            }

//            for (i in 0 until this.length()) {
//                val ef = getEFValue(i)
//                setEFValue(ef)
//            }

        } catch (e: JSONException) {
            Log.w(TAG, "updateLengthAreaVolumeAllFrame ${e}")

        }
    }


    fun getEFByVolumeAllFrame(): EFObject {

        var minV = 0F
        var maxV = 0F
        var esvIdx = -1
        var edvIdx = -1

        try {

            val annotation = getJSONArray(KEY_ANNOTATION)
            for (i in 0 until annotation.length()) {
                val o = annotation.getJSONObject(i)
                val volume = o.getDouble(VOLUME).toFloat()

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
            Log.w("getEFByVolumeAllFrame", "$minV $maxV $esvIdx $edvIdx $ef")
            return EFObject(indexEDV = edvIdx, indexESV = esvIdx, volumeEDV = maxV, volumeESV = minV, efValue = ef)

        } catch (e: JSONException) {
            Log.w(TAG, "getEFByVolumeAllFrame ${e}")
            return EFObject()

        }
    }

    fun getEFFromMinVMaxV(minV: Float, maxV: Float): Float {
        return if(abs(maxV - 0F) < EPS ) -1F else (maxV - minV) / maxV
    }


//    fun changeArea(frameIdx: Int, key: String, tags: JSONObject) {
//        if (key != EF_BOUNDARY) return
//        try {
//            val nColumn = if (tags.has(HEX_CODE_COLUMNS)) tags.getString(HEX_CODE_COLUMNS).toFloat() else 636.0F
//            val nRow  = if (tags.has(HEX_CODE_ROWS)) tags.getString(HEX_CODE_ROWS).toFloat() else 422.0F
//            val deltaX = if (tags.has(HEX_CODE_PHYSICAL_DELTA_X)) tags.getDouble(HEX_CODE_PHYSICAL_DELTA_X).toFloat() else 0.044318250253494909F
//            val deltaY = if (tags.has(HEX_CODE_PHYSICAL_DELTA_Y)) tags.getDouble(HEX_CODE_PHYSICAL_DELTA_Y).toFloat() else 0.044318250253494909F
//
//            val o = getJSONObject(frameIdx)
//            val boundary = o.getJSONArray(key)
//            if (boundary.length() >= 1) {
//                // chi tinh gia tri voi bounday[0]
//                val path = boundary.getJSONArray(0)
//                val length = getMeasureByKey(frameIdx, LENGTH)
//                val area = getAreaPath(path, deltaX, deltaY, nColumn, nRow)
////                Log.w(TAG, "L: $length, A: $area")
//                o.put(AREA, area)
//
//                val volume = if (abs(length - 0F) < EPS) 0F else (0.85F * area * area) / length / 1000.0F // mL
//                o.put(VOLUME, volume)
//
//                for (i in 0 until this.length()) {
////                    Log.w(TAG, "frameIdx info: ${i} L = ${getMeasureByKey(i, LENGTH)}, A = ${getMeasureByKey(i, AREA)}, V = ${getMeasureByKey(i, VOLUME)}")
//                    val ef = getEFValue(i)
////                    Log.w(TAG, "frameIdx: ${ef}")
//                    setEFValue(ef)
//                }
//            }
//
//        } catch (e: JSONException) {
//            Log.w(TAG, "changeArea ${key} ${frameIdx} ${e}")
//
//        }
//
//    }




}



class FrameAnnotation: JSONObject {

    constructor() : super() {

        drawTypes.forEach {
            put(it, JSONArray())
        }
        put(KEY_GLS_MANUAL, JSONArray())
        put(KEY_GLS_AUTO, JSONArray())

        put(IS_ESV, false)
        put(IS_EDV, false)
        put(IS_COPY, false)
        put(LENGTH, SPECIAL_VALUE)
        put(AREA, SPECIAL_VALUE)
        put(VOLUME, SPECIAL_VALUE)
        put(KEY_VOLUME_ESV, SPECIAL_VALUE)
        put(KEY_VOLUME_EDV, SPECIAL_VALUE)
        put(KEY_EF_MANUAL, SPECIAL_VALUE)
        put(KEY_EF_AUTO, SPECIAL_VALUE)

        put(KEY_INDEX_FRAME, SPECIAL_INDEX)
        put(KEY_INDEX_ESV, SPECIAL_INDEX)
        put(KEY_INDEX_EDV, SPECIAL_INDEX)

    }

    constructor(o: JSONObject) {

        drawTypes.forEach {
            put(it, try {o.getJSONArray(it)} catch (e: JSONException) { JSONArray() } )
        }
        put(KEY_GLS_AUTO, try {o.getJSONArray(KEY_GLS_AUTO)} catch (e: JSONException) { JSONArray() } )
        put(KEY_GLS_MANUAL, try {o.getJSONArray(KEY_GLS_MANUAL)} catch (e: JSONException) { JSONArray() } )

        put(IS_ESV, try {o.getBoolean(IS_ESV)} catch (e: JSONException) { false} )
        put(IS_EDV, try {o.getBoolean(IS_EDV)} catch (e: JSONException) { false} )
        put(IS_COPY, try {o.getBoolean(IS_COPY)} catch (e: JSONException) { false} )

        put(LENGTH, try {o.getDouble(LENGTH)} catch (e: JSONException) { SPECIAL_VALUE} )
        put(AREA, try {o.getDouble(AREA)} catch (e: JSONException) { SPECIAL_VALUE} )
        put(VOLUME, try {o.getDouble(VOLUME)} catch (e: JSONException) { SPECIAL_VALUE} )

        put(KEY_VOLUME_ESV, try {o.getDouble(KEY_VOLUME_ESV)} catch (e: JSONException) { SPECIAL_VALUE} )
        put(KEY_VOLUME_EDV, try {o.getDouble(KEY_VOLUME_EDV)} catch (e: JSONException) { SPECIAL_VALUE} )
        put(KEY_EF_MANUAL, try {o.getDouble(KEY_EF_MANUAL)} catch (e: JSONException) { SPECIAL_VALUE} )
        put(KEY_EF_AUTO, try {o.getDouble(KEY_EF_AUTO)} catch (e: JSONException) { SPECIAL_VALUE} )

        put(KEY_INDEX_FRAME, try {o.getInt(KEY_INDEX_FRAME)} catch (e: JSONException) { SPECIAL_INDEX} )
        put(KEY_INDEX_ESV, try {o.getInt(KEY_INDEX_ESV)} catch (e: JSONException) { SPECIAL_INDEX} )
        put(KEY_INDEX_EDV, try {o.getInt(KEY_INDEX_EDV)} catch (e: JSONException) { SPECIAL_INDEX} )

    }

    companion object {

        const val KEY_POINT = "point"
        const val KEY_BOUNDARY = "boundary"

        const val EF_POINT = "ef_point"
        const val EF_BOUNDARY = "ef_boundary"
        const val GLS_POINT = "gls_point"
        const val GLS_BOUNDARY = "gls_boundary"

        const val MEASURE_LENGTH = "measure_length"
        const val MEASURE_AREA = "measure_area"

        val drawTypes: Array<String> = arrayOf(EF_POINT, EF_BOUNDARY, GLS_POINT, GLS_BOUNDARY, MEASURE_LENGTH, MEASURE_AREA)

        const val IS_ESV = "IS_ESV"
        const val IS_EDV = "IS_EDV"
        const val IS_COPY = "IS_COPY"

        const val LENGTH = "Length"
        const val AREA = "Area"
        const val VOLUME = "Volume"

        const val KEY_INDEX_FRAME = "FrameID"
        const val KEY_INDEX_ESV = "FrameESV"
        const val KEY_INDEX_EDV = "FrameEDV"
        const val KEY_VOLUME_ESV = "VolumeESV"
        const val KEY_VOLUME_EDV = "VolumeEDV"
        const val KEY_EF_MANUAL = "EF_MANUAL"
        const val KEY_EF_AUTO = "EF_AUTO"
        const val KEY_GLS_MANUAL = "GLS_MANUAL"
        const val KEY_GLS_AUTO = "GLS_AUTO"

        const val SPECIAL_VALUE = 0F
        const val SPECIAL_INDEX = -1

    }


}

class DicomDiagnosis: JSONObject {

    companion object {
        const val TAG = "DicomDiagnosis"
        const val TOOL_NAME = "dicomDiagnosis"
        const val CHAMBER = "chamber"
        const val CHAMBER_IDX = "chamber_idx"
        const val NOT_STANDARD = "not_standard"
        const val POINTS = "points"
        const val LAD = "lad"
        const val RCA = "rca"
        const val LCX = "lcx"
        const val NOTE = "note"

        const val ANORMAL_hyperactivity = 1
        const val NORMAL_motion = 2
        const val ANORMAL_post_systolic_contraction = 3
        const val ANORMAL_hypokinetic_motion = 4
        const val ANORMAL_akinetic_motion = 5
        const val ANORMAL_dyskinetic_motion = 6
        const val ANORMAL_paradoxical_motion = 7
        const val ANORMAL_dyssynchronized_motion = 8



        fun getChamberName(index: Int) : String {
            return when (index) {
                0 -> "2C"
                1 -> "3C"
                2 -> "4C"
                3 -> "PTS_L"
                4 -> "PTS_S"
                5 -> "NO"
                else -> "LABEL"
            }
        }

        fun getChamberIdxFromName(name: String) : Int {
            return when (name) {
                "2C" -> 0
                "3C" -> 1
                "4C" -> 2
                "PTS_L" -> 3
                "PTS_S" -> 4
                "NO" -> 5
                else -> -1
            }
        }

        fun getPointType(point: JSONObject): Int {
            try {
                return point.getInt("type")

            } catch (e: JSONException) {
                Log.w(TAG, "getPointType ${point} ${e}")
                return -1
            }
        }

    }

    constructor() : super() {
        put(CHAMBER, getChamberName(-1))
        put(CHAMBER_IDX, -1)
        put(NOT_STANDARD, false)
        put(LAD, false)
        put(RCA, false)
        put(LCX, false)
        put(POINTS, JSONArray())
        put(NOTE, "")
    }
    constructor(s: String) : super(s) {
        if (!this.has(CHAMBER)) put(CHAMBER, getChamberName(-1))
        if (!this.has(CHAMBER_IDX)) put(CHAMBER_IDX, -1)
        if (!this.has(NOT_STANDARD)) put(NOT_STANDARD, false)
        if (!this.has(LAD)) put(LAD, false)
        if (!this.has(RCA)) put(RCA, false)
        if (!this.has(LCX)) put(LCX, false)
        if (!this.has(POINTS)) put(POINTS, JSONArray())
        if (!this.has(NOTE)) put(NOTE, "")
    }

    constructor(s: JSONObject){
        put(CHAMBER,try {s.getString(CHAMBER)} catch (e: Exception) {getChamberName(-1)} )
        put(CHAMBER_IDX, try{getChamberIdxFromName(s.getString(CHAMBER))} catch (e: Exception) {getChamberIdxFromName("")}  )

        put(NOT_STANDARD,try {s.getBoolean(NOT_STANDARD)} catch (e: Exception) {false} )
        put(LAD,try {s.getBoolean(LAD)} catch (e: Exception) {false} )
        put(RCA,try {s.getBoolean(RCA)} catch (e: Exception) {false} )
        put(LCX,try {s.getBoolean(LCX)} catch (e: Exception) {false} )
        put(POINTS,try {s.getJSONArray(POINTS)} catch (e: Exception) {JSONArray()} )
        put(NOTE,try {s.getString(NOTE)} catch (e: Exception) {""} )


    }

    var points : JSONArray
        get() = getJSONArray(POINTS)
        set(value) { put(POINTS, value) }

    val nPoints: Int get() = points.length()

    fun getPoint(i: Int): JSONObject {
        try {
            return points.getJSONObject(i)
        } catch (e: JSONException) {
            Log.w(TAG, "getPoint ${i} ${e}")
            return JSONObject()
        }
    }
    fun clearPoints() {
        points = JSONArray()
    }
    fun addPoint(ix: Float, iy: Float, atype: Int) {
        val o = JSONObject()
        o.put("x", ix)
        o.put("y", iy)
        o.put("type", atype)
        points.put(o)
    }


    var chamberIdx: Int
        get() {
            return try {
                getInt(CHAMBER_IDX)
            } catch (e: JSONException) {
                Log.w(TAG, "GET chamberIdx ${e}")
                -1
            }
        }
        set(value) {
            put(CHAMBER_IDX, value)
            put(CHAMBER, getChamberName(value))
        }

    val chamber: String
        get() {
            return try {
                getString(CHAMBER)
            } catch (e: JSONException) {
                Log.w(TAG, "GET chamber ${e}")
                "LABEL"
            }
        }


    var lad : Boolean
        get() {
            return try {
                getBoolean(LAD)
            } catch (e: JSONException) {
                Log.w(TAG, "GET lad ${e}")
                false
            }
        }
        set(value) { put(LAD, value) }

    var lcx : Boolean
        get() {
            return  try {
                getBoolean(LCX)
            } catch (e: JSONException) {
                Log.w(TAG, "GET lcx ${e}")
                false
            }
        }
        set(value) { put(LCX, value) }

    var rca : Boolean
        get() {
            return  try {
                getBoolean(RCA)
            } catch (e: JSONException) {
                Log.w(TAG, "GET rca ${e}")
                false
            }
        }
        set(value) { put(RCA, value) }

    var isNotStandardImage : Boolean
        get() {
            return  try {
                getBoolean(NOT_STANDARD)
            } catch (e: JSONException) {
                Log.w(TAG, "GET isNotStandardImage ${e}")
                false
            }
        }
        set(value) { put(NOT_STANDARD, value) }

    var note: String
        get() {
            return  try {
                getString(NOTE)
            } catch (e: JSONException) {
                Log.w(TAG, "GET note ${e}")
                ""
            }
        }
        set(value) { put(NOTE, value) }
}

