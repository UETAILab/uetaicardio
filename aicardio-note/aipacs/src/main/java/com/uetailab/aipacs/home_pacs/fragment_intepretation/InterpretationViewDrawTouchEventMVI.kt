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

import android.annotation.SuppressLint
import android.graphics.*
import android.util.Log
import com.uetailab.aipacs.R
import com.uetailab.aipacs.home_pacs.fragment_intepretation.DicomInterpretation.Companion.EPS
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.min

class InterpretationViewDrawTouchEventMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: InterpretationViewDrawTouchEventMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: InterpretationViewDrawTouchEventMVI()
                        .also { instance = it }
            }

        const val TAG = "InterpretationViewDrawTouchEventMVI"

        const val TYPE_POINT_EF = "TYPE_POINT_EF"
        const val TYPE_POINT_GLS = "TYPE_POINT_GLS"
        const val TYPE_POINT_MEASURE_LENGTH = "TYPE_POINT_MEASURE_LENGTH"


        const val TYPE_BOUNDARY_EF = "TYPE_BOUNDARY_EF"
        const val TYPE_BOUNDARY_GLS = "TYPE_BOUNDARY_GLS"
        const val TYPE_BOUNDARY_MEASURE_AREA = "TYPE_BOUNDARY_MEASURE_AREA"

        const val TYPE_DRAW_MANUAL = "TYPE_DRAW_MANUAL"
        const val TYPE_DRAW_AUTO = "TYPE_DRAW_AUTO"

        fun process(interpretationViewVM: InterpretationViewVM, InterpretationViewEvent: InterpretationViewEvent) {
            getInstance().process(interpretationViewVM, InterpretationViewEvent)
        }

        fun renderViewState(interpretationViewFragment: InterpretationViewFragment, viewState: InterpretationViewState) {
            getInstance().renderViewState(interpretationViewFragment, viewState)
        }

        fun renderViewEffect(interpretationViewFragment: InterpretationViewFragment, viewEffect: InterpretationViewEffect) {
            getInstance().renderViewEffect(interpretationViewFragment, viewEffect)
        }

        // Manual EF, Manual GLS, Auto EF, Auto GLS
        // EF: point, boundary, auto, manual

        fun getPaintDrawPoint(typePoint: String, typeDraw: String) : Paint {
            val paint = Paint()

            paint.color = when(typePoint) {
                TYPE_POINT_GLS -> {
//                    Log.w(TAG, "GO TO GLS POINT")
                    when (typeDraw) {
                        TYPE_DRAW_MANUAL -> Color.RED
                        else -> Color.YELLOW
                    }
                }

                TYPE_POINT_EF -> {
                    when (typeDraw) {
                        TYPE_DRAW_MANUAL ->  Color.GREEN
                        else -> Color.BLUE
                    }
                }
                TYPE_POINT_MEASURE_LENGTH -> Color.MAGENTA
                else -> R.color.pink_primary_dark // TYPE_POINT_MEASURE_AREA
            }
            paint.textSize = 30F
            return paint
        }
        fun getPaintDrawLine(typeBoundary: String=TYPE_BOUNDARY_GLS, typeDraw: String= TYPE_DRAW_MANUAL) : Paint {
            val paint = Paint()
            paint.color = when(typeBoundary) {

                TYPE_BOUNDARY_GLS -> {
//                    Log.w(TAG, "GO TO GLS POINT")
                    when (typeDraw) {
                        TYPE_DRAW_MANUAL -> Color.RED
                        else -> Color.YELLOW
//                        else -> 0xFF800000.toInt() // maroon
                    }
                }

                TYPE_BOUNDARY_EF -> {

                    when (typeDraw) {
                        TYPE_DRAW_MANUAL ->  Color.GREEN
                        else -> Color.BLUE
//                        else -> 0xFF808000.toInt()
                    }
                }

                TYPE_BOUNDARY_MEASURE_AREA -> Color.MAGENTA

                else -> R.color.pink_primary_dark // TYPE_POINT_MEASURE_AREA
            }

            paint.strokeWidth = 3.0F
            paint.textSize = 30F
            paint.style = Paint.Style.STROKE
            if (typeDraw == TYPE_DRAW_AUTO) paint.pathEffect = DashPathEffect(floatArrayOf(10F, 10F, 10F, 10F), 0F)
            return paint
        }

        fun getPaintDrawPolygon(typeBoundary: String, typeDraw: String) : Paint {
            val paint = Paint()
            paint.color = when(typeBoundary) {

                TYPE_BOUNDARY_GLS -> {
//                    Log.w(TAG, "GO TO GLS POINT")
                    when (typeDraw) {
                        TYPE_DRAW_MANUAL -> Color.YELLOW
                        else -> Color.RED
//                        else -> 0xFF800000.toInt() // maroon
                    }
                }

                TYPE_BOUNDARY_EF -> {

                    when (typeDraw) {
                        TYPE_DRAW_MANUAL ->  Color.BLUE
                        else -> Color.GREEN
//                        else -> 0xFF808000.toInt()
                    }
                }

                TYPE_BOUNDARY_MEASURE_AREA -> Color.MAGENTA

                else -> R.color.pink_primary_dark // TYPE_POINT_MEASURE_AREA
            }
            paint.strokeWidth = 3.0F
            paint.alpha = 30
            paint.textSize = 30F
            return paint
        }
//        DicomDiagnosis.ANORMAL_hyperactivity ->
//        DicomDiagnosis.NORMAL_motion -> R.color.colorGrey
//        DicomDiagnosis.ANORMAL_post_systolic_contraction -> R.color.colorPurple
//        DicomDiagnosis.ANORMAL_hypokinetic_motion -> R.color.colorGreen
//        DicomDiagnosis.ANORMAL_akinetic_motion -> R.color.colorYellow
//        DicomDiagnosis.ANORMAL_dyskinetic_motion -> R.color.colorRed
//        DicomDiagnosis.ANORMAL_paradoxical_motion -> R.color.colorBrown
//        DicomDiagnosis.ANORMAL_dyssynchronized_motion -> R.color.colorBlack
//
        val listPaintSectors = arrayListOf<Int>(Color.BLUE, Color.CYAN, Color.GREEN, Color.LTGRAY, Color.MAGENTA, Color.RED)

        @SuppressLint("ResourceAsColor")
        fun getPaintDrawSector(idSector: Int) : Paint {
            val paint = Paint()
            paint.color = listPaintSectors.get(idSector)
//            Log.w(TAG, "getPaintDrawSector: ${listPaintSectors.get(idSector)}")
            paint.strokeWidth = 5.0F
            paint.alpha = 200
            return paint
        }

        fun getPaintDrawText(): Paint {
            val textPaint = Paint()
            textPaint.color = Color.CYAN
            textPaint.strokeWidth = 3F
            textPaint.textSize = 30F
            return textPaint
        }
        fun drawBoundary(viewState: InterpretationViewState, view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?, boundary: JSONArray, paint: Paint, drawTextArea: Boolean=false): Float {
            try {
                return drawMultiPolygon(viewState, view, canvas, boundary, paint, drawTextArea)

            } catch (e: JSONException) {
                Log.w(TAG, "drawBoundary ${e}")
                return 0F
            }
        }

        fun drawMultiPolygon(viewState: InterpretationViewState, view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?,
                             boundary: JSONArray, paint: Paint, drawTextArea: Boolean=false) : Float{
            try {
                var result = 0F
                if (boundary.length() > 0){
                    val path = Path()
                    val p = view.getScreenCoordinate(boundary.getJSONArray(0).getJSONObject(0))
                    path.moveTo(p[0], p[1])
                    repeat(boundary.length()){
                        val paths = boundary.getJSONArray(it)
                        val n = paths.length()

                        for (i in 0 until n){
                            val point = view.getScreenCoordinate(paths.getJSONObject(i))
                            path.lineTo(point[0], point[1])
                        }

                        if (drawTextArea && n > 1) {
                            //
                            result = getAreaPath(paths, viewState.deltaX, viewState.deltaY, viewState.nColumn, viewState.nRow)
                            /** draw text is area of this boundary */
                            drawText(
                                view,
                                canvas,
                                " %.1f cm2".format(result / 100.0F),
                                paths.getJSONObject(0),
                                getPaintDrawText()
                            )

                        }


                    }
                    canvas?.drawPath(path, paint)

                }
                return result
            } catch (e: JSONException) {
                Log.w(TAG, "drawPolygon ${e}")
                return 0F
            }

        }

        fun drawPolygon(viewState: InterpretationViewState, view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?,
                             boundary: JSONArray, paint: Paint, drawTextArea: Boolean=false) : Float{
            try {
                var result = 0F
                    val path = Path()
                    val p = view.getScreenCoordinate(boundary.getJSONObject(0))
                    path.moveTo(p[0], p[1])

                    val paths = boundary
                    val n = paths.length()

                    for (i in 0 until n){
                        val point = view.getScreenCoordinate(paths.getJSONObject(i))
                        path.lineTo(point[0], point[1])
                    }

//                    if (drawTextArea && n > 1) {
//                        //
//                        result = getAreaPath(paths, viewState.deltaX, viewState.deltaY, viewState.nColumn, viewState.nRow)
//                        /** draw text is area of this boundary */
//                        drawText(
//                            view,
//                            canvas,
//                            " %.1f cm2".format(result / 100.0F),
//                            paths.getJSONObject(0),
//                            getPaintDrawText()
//                        )
//
//                    }
                    canvas?.drawPath(path, paint)

                return result
            } catch (e: JSONException) {
                Log.w(TAG, "drawPolygon ${e}")
                return 0F
            }

        }



        fun drawLine(view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?, p1: JSONObject?, p2: JSONObject?, paint: Paint) {
            try {
                if (p1 == null || p2 == null)
                    return
                view.bitmap?.let {
                    val point1 = view.getScreenCoordinate(p1)
                    val point2 = view.getScreenCoordinate(p2)
                    canvas?.drawLine(point1[0], point1[1], point2[0], point2[1], paint)

                }
            } catch (e: JSONException) {
                Log.w(TAG, "drawLine ${e}")

            }

        }

        fun drawPointsDialog(viewState: InterpretationViewState, view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?, points: JSONArray, paint: Paint ): Float {
            try {
                var result = 0F

//                repeat(points.length()) {
//
//                    val point = view.getScreenCoordinate(points.getJSONObject(it))
//                    canvas?.drawCircle(point[0], point[1], 10.0F, paint)
//
//                    canvas?.drawText("${it + 1}" , point[0] + 15, point[1] + 15, paint)
//
//
//                }

                if (points.length() >= 7 && points.length() % 2 == 1) {
                    val np = points.length()
//                    Log.w(TAG, "Draw point: ${np/2} ${np}")
                    val p0 = points.getJSONObject(0)
                    val p3 = points.getJSONObject(np / 2)
                    val p6 = points.getJSONObject(np -1)



                    val pMid06 = getMiddelPoint(p0, p6)

//                    length =  getLengthPoint(p3, pMid06, deltaX, deltaY, nColumn, nRow)

                    val s1 = view.getScreenCoordinate(p3)
                    val s2 = view.getScreenCoordinate(pMid06)

//                    canvas?.drawLine(s1[0], s1[1], s2[0], s2[1], getPaintDrawLine())

                    result = getLengthPoint(
                        p3,
                        pMid06,
                        viewState.deltaX,
                        viewState.deltaY,
                        viewState.nColumn,
                        viewState.nRow
                    )

//                    canvas?.drawCircle(s2[0], s2[1], 10.0F, paint)

//                    canvas?.drawText(" %.1f mm".format(result), (s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2,
//                        getPaintDrawText()
//                    )

                }
                return result
            } catch (e: JSONException) {
                Log.w(TAG, "drawPoints ${e}")
                return 0F
            }

        }

        fun drawPoints(viewState: InterpretationViewState, view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?,
                       points: JSONArray, paint: Paint, isViewLength: Boolean=false): Float {
            try {
                var result = 0F

                repeat(points.length()) {

                    val point = view.getScreenCoordinate(points.getJSONObject(it))
                    canvas?.drawCircle(point[0], point[1], 10.0F, paint)

                    canvas?.drawText("${it + 1}" , point[0] + 15, point[1] + 15, paint)


                }

                if (points.length() >= 7 && points.length() % 2 == 1) {
                    val np = points.length()
//                    Log.w(TAG, "Draw point: ${np/2} ${np}")
                    val p0 = points.getJSONObject(0)
                    val p3 = points.getJSONObject(np / 2)
                    val p6 = points.getJSONObject(np -1)



                    val pMid06 = getMiddelPoint(p0, p6)

//                    length =  getLengthPoint(p3, pMid06, deltaX, deltaY, nColumn, nRow)

                    val s1 = view.getScreenCoordinate(p3)
                    val s2 = view.getScreenCoordinate(pMid06)

//                    canvas?.drawLine(s1[0], s1[1], s2[0], s2[1], getPaintDrawLine())

                    result = getLengthPoint(
                        p3,
                        pMid06,
                        viewState.deltaX,
                        viewState.deltaY,
                        viewState.nColumn,
                        viewState.nRow
                    )

//                    canvas?.drawCircle(s2[0], s2[1], 10.0F, paint)

                    if (isViewLength) canvas?.drawText(" %.1f mm".format(result), (s1[0] + s2[0]) / 2, (s1[1] + s2[1]) / 2,
                        getPaintDrawText()
                    )

                }
                return result
            } catch (e: JSONException) {
                Log.w(TAG, "drawPoints ${e}")
                return 0F
            }

        }

        fun drawPoint(view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?, p: JSONObject, paint: Paint) {
            val point = view.getScreenCoordinate(p)
            canvas?.drawCircle(point[0], point[1], 2.0F, paint)
        }

        fun drawText(view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?, text: String, p: JSONObject, paint: Paint) {
            val point = view.getScreenCoordinate(p)
            canvas?.drawText(text, point[0], point[1], paint)
        }


        fun drawPointKnot(view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?, knots: JSONArray, paint: Paint){
            try {
                repeat(knots.length()){i->
                    val knot = knots.getJSONArray(i)
                    repeat(knot.length()){j->
                        val point = view.getScreenCoordinate(knot.getJSONObject(j))
                        canvas?.drawCircle(point[0], point[1], 4F, paint)
                    }
                }

            } catch (e: JSONException) {
                Log.w(TAG, "drawPointKnot  ${e}")

            }

        }

        fun drawBoundaryAreaTool(viewState: InterpretationViewState, view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?,
                                 boundary: JSONArray, paintText: Paint, paintLine: Paint, paintPoint: Paint, isVisibleAreaText: Boolean =false) {
            canvas?.let {
                try {
                    repeat(boundary.length()) {
                        val paths = boundary.getJSONArray(it)
                        val n = paths.length()
                        /** drawLine from point[i] to point[i + 1] */
                        for (i in 0..n-2) {
                            drawLine(view, canvas, paths.getJSONObject(i), paths.getJSONObject(i + 1), paintLine)
                        }
                        if (n >= 1) {
                            /** drawLine point[n - 1] to point[0] */
                            drawLine(view, canvas, paths.getJSONObject(n - 1), paths.getJSONObject(0), paintLine)

                            /** draw point[0] */
                            drawPoint(view, canvas, paths.getJSONObject(0), paintPoint)

                            /** draw point[n - 1] */
                            drawPoint(view, canvas, paths.getJSONObject(n - 1), paintPoint)

                        }
                        // isVisibleAreaText
                        if (n > 1) {
                            /** draw text is area of this boundary */
                            drawText(
                                view,
                                canvas,
                                getAreaPathText(paths, viewState.deltaX, viewState.deltaY, viewState.nColumn, viewState.nRow),
                                paths.getJSONObject(0),
                                paintText
                            )

                        }
                    }

                } catch (e: JSONException) {
                    Log.w(TAG, "drawBoundaryAreaTool ${e}")

                }

            }
        }

        fun drawPointLengthTool(viewState: InterpretationViewState, view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?, points: JSONArray, paintText: Paint, paintLine: Paint, paintPoint: Paint) {
            canvas?.let {
                try {
                    for(i in 0 until points.length()) {
                        val p1 = points.getJSONObject(i)
                        val s1 = view.getScreenCoordinate(p1)

                        if (i % 2 == 0 && i < points.length()-1) {
                            val p2 = points.getJSONObject(i + 1)
                            val s2 = view.getScreenCoordinate(p2)

                            canvas.drawLine(s1[0], s1[1], s2[0], s2[1], paintLine)

                            canvas.drawText(
                                getLengthPointText(
                                    p1,
                                    p2,
                                    viewState.deltaX,
                                    viewState.deltaY,
                                    viewState.nColumn,
                                    viewState.nRow
                                ), s2[0], s2[1],
                                paintText
                            )

                        }

                        canvas.drawCircle(s1[0], s1[1], 6.0F, paintPoint)
                    }
                } catch (e: JSONException) {
                    Log.w(TAG, "drawPointLengthTool ${e}")

                }
            }

        }




        fun getAreaPathText(path: JSONArray, deltaX: Float, deltaY: Float, nColumn: Float, nRow: Float): String {
            val area = getAreaPath(path, deltaX, deltaY, nColumn, nRow)
            return " %.1f cm2".format(area / 100.0F)
        }

        fun getLengthPointText(p1: JSONObject, p2: JSONObject, deltaX: Float, deltaY: Float, nColumn: Float, nRow: Float): String {
            val l= getLengthPoint(p1, p2, deltaX, deltaY, nColumn, nRow)
            return " %.1f mm".format(l)
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

    }

    private fun renderViewEffect(interpretationViewFragment: InterpretationViewFragment, viewEffect: InterpretationViewEffect) {

    }

    private fun renderViewState(interpretationViewFragment: InterpretationViewFragment, viewState: InterpretationViewState) {

    }

    fun process(interpretationViewVM: InterpretationViewVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {
            is InterpretationViewEvent.RenderTouchDraw -> {
                interpretationViewVM.viewStates().value?.let {
                    interpretationViewVM.reduce(RenderTouchDrawReducer(interpretationViewVM, it, interpretationViewEvent))
                }
            }
        }
    }


    inner class RenderTouchDrawReducer(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.RenderTouchDraw) : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {


            viewModel.viewStates().value?.let {
                if (viewEvent.enableAutoDraw) drawByType(it, viewModel, viewEvent, TYPE_DRAW_AUTO)
                if (viewEvent.enableManualDraw) drawByType(it, viewModel, viewEvent, TYPE_DRAW_MANUAL)

//                drawAutoBySector(it, viewModel, viewEvent, TYPE_DRAW_AUTO)
//                drawAutoBySector(it, viewModel, viewEvent, TYPE_DRAW_MANUAL)

            }

            return InterpretationViewObject()
        }


    }

    fun drawAutoBySector(viewState: InterpretationViewState, viewModel: InterpretationViewVM, viewEvent: InterpretationViewEvent.RenderTouchDraw, typeDraw: String) {
        val numFrame = viewModel.numFrame
        val currentFrameIndex = viewModel.getCurrentFrameIndex()

        val view = viewEvent.view
        val canvas = viewEvent.canvas



        view.bitmap?.let {
            val annotation: DicomInterpretation = if (typeDraw == TYPE_DRAW_MANUAL) viewState.dicomInterpretation else viewState.machineInterpretation

            if (numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame) {

                val efBoundary = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.EF_BOUNDARY)
                val glsBoundary = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.GLS_BOUNDARY)

                val sectors = convertPointToSector(efBoundary, glsBoundary)
                repeat(sectors.length()) { i ->
                    val paint = getPaintDrawSector(idSector = i)
//                    if (i == 0)
                    drawPolygon(viewState, view, canvas, sectors.getJSONArray(i), paint = paint )

                }
            }
        }
    }

    fun convertPointToSector(efBoundary: JSONArray, glsBoundary: JSONArray, numSector: Int=6): JSONArray {
        val result = JSONArray()

        try {
            val b1 = efBoundary.getJSONArray(0)
            val b2 = glsBoundary.getJSONArray(0)
            val n1 = b1.length()
            val n2 = b2.length()
            val sn1 = n1 / numSector
            val sn2 = n2 / numSector
            for (i in 0..numSector - 1) {
                val o = JSONArray()
                val startIDN1 = i * sn1
                val endIDN1 = (i + 1) * sn1 - 1
                Log.w(TAG, "convertPointToSector GET POINT ${i} start: ${startIDN1} end: ${endIDN1} n1: ${n1}")
                for (j in startIDN1..min(endIDN1, n1)) {
//                val o1 = JSONObject()
//                o1.put('x', b1.get(j))
                    o.put(b1.get(j))
                }

//                for (j in i * sn2..min( (i + 1) * sn2 - 1, n2 )) {
//                    o.put(b2.get(j))
//                }
                for (j in min( (i + 1) * sn2 - 1, n2 ) downTo i * sn2 ) {
                    o.put(b2.get(j))
                }
                result.put(o)
            }
        } catch (e: JSONException) {
            Log.w(TAG, "convertPointToSector ${e}")
            0F
        }

        return result
    }


    fun drawByType(viewState: InterpretationViewState, viewModel: InterpretationViewVM, viewEvent: InterpretationViewEvent.RenderTouchDraw, typeDraw: String) {

        val numFrame = viewModel.numFrame
        val currentFrameIndex = viewModel.getCurrentFrameIndex()

        val view = viewEvent.view
        val canvas = viewEvent.canvas


        val currentInterpretationToolClick = viewModel.getCurrentInterpretationToolClick()
        val isLongClicked = currentInterpretationToolClick.isLongClicked
        val isGls = viewModel.getIsGls()


        view.bitmap?.let {
            val annotation: DicomInterpretation = if (typeDraw == TYPE_DRAW_MANUAL) viewState.dicomInterpretation else viewState.machineInterpretation
            if (numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame) {
                // draw for ef (point, boundary)
                val efPoints = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.EF_POINT)
                val efBoundary = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.EF_BOUNDARY)
//                Log.w(TAG, "efPoints: ${efPoints}")
                val length = drawPoints(viewState, view, canvas, efPoints,
                    paint = getPaintDrawPoint(typePoint = TYPE_POINT_EF, typeDraw = typeDraw), isViewLength = true )
                val area = drawBoundary(viewState, view, canvas, efBoundary, paint = getPaintDrawLine(typeBoundary = TYPE_BOUNDARY_EF, typeDraw = typeDraw), drawTextArea = true )
                drawMultiPolygon(viewState, view, canvas, efBoundary, paint = getPaintDrawPolygon(typeBoundary = TYPE_BOUNDARY_EF, typeDraw = typeDraw) )

                val volume = if (abs(length - 0F) < EPS) 0F else (0.85F * area * area) / length / 1000.0F // mL
                if (volume > 0) {
                    // draw volume at index 2
                    val nPoint = efPoints.length()
                    if (nPoint >= 7) {
                        val point = view.getScreenCoordinate(efPoints.getJSONObject(  nPoint - 1))
                        canvas?.drawText(" %.2f mL".format(volume) , point[0] + 30, point[1] + 30, getPaintDrawText())
                    }

                }

                // mod edit boundary ef
                if (currentInterpretationToolClick is InterpretationViewTool.OnClickDrawBoundary && isLongClicked && isGls == false)
                    drawPointKnot( view, canvas, knots = annotation.getKnots(currentFrameIndex, key = FrameAnnotation.EF_BOUNDARY), paint = getPaintDrawLine(typeBoundary = TYPE_BOUNDARY_GLS, typeDraw = typeDraw) )

                // draw for gls (point, boundary)
                val glsPoints = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.GLS_POINT)
//                Log.w(TAG, "glsPoints: ${glsPoints}")
                val glsBoundary = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.GLS_BOUNDARY)

                drawPoints(viewState, view, canvas, glsPoints, getPaintDrawPoint(typePoint = TYPE_POINT_GLS, typeDraw = typeDraw) )
                drawBoundary(viewState, view, canvas, glsBoundary, paint = getPaintDrawLine(typeBoundary = TYPE_BOUNDARY_GLS, typeDraw = typeDraw), drawTextArea = false)
                drawMultiPolygon(viewState, view, canvas, glsBoundary, paint = getPaintDrawPolygon(typeBoundary = TYPE_BOUNDARY_GLS, typeDraw = typeDraw) )

                // mod edit boundary gls
                if (currentInterpretationToolClick is InterpretationViewTool.OnClickDrawBoundary && isLongClicked && isGls == true)
                    drawPointKnot( view, canvas, knots = annotation.getKnots(currentFrameIndex, key = FrameAnnotation.GLS_BOUNDARY), paint = getPaintDrawLine(typeBoundary = TYPE_BOUNDARY_GLS, typeDraw = typeDraw) )

                // mod draw length
                val pointLengthTool = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.MEASURE_LENGTH)
                drawPointLengthTool( viewState, view, canvas, pointLengthTool, paintText = getPaintDrawText(), paintLine = getPaintDrawLine( typeBoundary = TYPE_BOUNDARY_MEASURE_AREA, typeDraw = typeDraw ), paintPoint = getPaintDrawPoint(typePoint = TYPE_POINT_MEASURE_LENGTH, typeDraw = typeDraw) )

                // mod draw area
                val boundaryAreaTool = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.MEASURE_AREA)
                drawBoundaryAreaTool( viewState, view, canvas, boundaryAreaTool, paintText = getPaintDrawText(), paintLine = getPaintDrawLine( typeBoundary = TYPE_BOUNDARY_MEASURE_AREA, typeDraw = typeDraw ), paintPoint = getPaintDrawPoint(typePoint = TYPE_POINT_MEASURE_LENGTH, typeDraw = typeDraw) )
            }
        }
    }


}
