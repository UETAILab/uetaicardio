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

package com.ailab.aicardio.annotationscreen

import android.graphics.*
import android.util.Log
import com.ailab.aicardio.R
import com.ailab.aicardio.TAG_LONG_CLICKED
import com.ailab.aicardio.annotationscreen.views.DrawCanvasView
import com.ailab.aicardio.getAreaPath
import com.ailab.aicardio.getLengthPoint
import com.ailab.aicardio.repository.AnnotationStateEffectObject
import com.ailab.aicardio.repository.DicomAnnotation
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

class RenderDrawMVI {

    companion object {
        const val TAG = "RenderDrawUniDirectionMVI"
        // For Singleton instantiation
        @Volatile
        private var instance: RenderDrawMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: RenderDrawMVI()
                        .also { instance = it }
            }

        fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
            getInstance()
                .process(annotationActVM, annotationViewEvent)
        }

        fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
            getInstance()
                .renderViewState(annotationActivity, viewState)
        }

        fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
            getInstance()
                .renderViewEffect(annotationActivity, viewEffect)
        }

        const val TYPE_POINT_EF = "TYPE_POINT_EF"
        const val TYPE_POINT_GLS = "TYPE_POINT_GLS"
        const val TYPE_POINT_MEASURE_LENGTH = "TYPE_POINT_MEASURE_LENGTH"


        const val TYPE_BOUNDARY_EF = "TYPE_BOUNDARY_EF"
        const val TYPE_BOUNDARY_GLS = "TYPE_BOUNDARY_GLS"
        const val TYPE_BOUNDARY_MEASURE_AREA = "TYPE_BOUNDARY_MEASURE_AREA"

        const val TYPE_DRAW_MANUAL = "TYPE_DRAW_MANUAL"
        const val TYPE_DRAW_AUTO = "TYPE_DRAW_AUTO"

//        fun getPaintDrawPoint(typePoint: String, typeDraw: String = "") : Paint {
//            val paint = Paint()
//            paint.color = when(typePoint) {
//                ProcessRenderDrawReducer.TYPE_POINT_GLS -> Color.RED
//                ProcessRenderDrawReducer.TYPE_POINT_EF -> Color.GREEN
//                ProcessRenderDrawReducer.TYPE_POINT_MEASURE_LENGTH -> Color.MAGENTA
//                else -> Color.GREEN
//            }
////            paint.color = if (isGls) Color.RED else Color.GREEN
////            paint.strokeWidth = 1.5F
//            paint.textSize = 30F
//            return paint
//        }

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
//                        else -> 0xFF800000.toInt() // maroon
                    }
                }

                TYPE_POINT_EF -> {

                    when (typeDraw) {
                        TYPE_DRAW_MANUAL ->  Color.GREEN
                        else -> Color.BLUE
//                        else -> 0xFF808000.toInt()
                    }
                }

                TYPE_POINT_MEASURE_LENGTH -> Color.MAGENTA

                else -> R.color.pink_primary_dark // TYPE_POINT_MEASURE_AREA
            }
//            paint.color = -0xff7f80 // 0xFF008080
//            paint.color = 0xFFC0C0C0.toInt() // 0xFFC0C0C0
//            0xFFC0C0C0
//            paint.color = getResources().getColor(R.color.magenta)
            paint.textSize = 30F
            return paint
        }

        fun getPaintDrawLine(typeBoundary: String, typeDraw: String) : Paint {
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
            paint.alpha = 50
            paint.textSize = 30F
            return paint
        }
        fun getPaintDrawText(): Paint {
            val textPaint = Paint()
            textPaint.color = Color.CYAN
            textPaint.strokeWidth = 3F
            textPaint.textSize = 30F
            return textPaint
        }
        fun drawBoundary(view: DrawCanvasView, canvas: Canvas?, boundary: JSONArray, paint: Paint) {
            try {
//                repeat(boundary.length()) {
//                    val path = boundary.getJSONArray(it)
//                    val n = path.length()
//                    for (i in 0 until n-1) {
//                        drawLine(view, canvas, path.getJSONObject(i), path.getJSONObject(i + 1), paint)
//                    }
//                    drawLine(view, canvas, path.getJSONObject(n-1), path.getJSONObject(0), paint)
//
//                }
                drawPolygon(view, canvas, boundary, paint)

            } catch (e: JSONException) {
                Log.w(TAG, "drawBoundary ${e}")

            }

        }

        fun drawPolygon(
            view: DrawCanvasView,
            canvas: Canvas?,
            boundary: JSONArray,
            paint: Paint
        ){
            try {
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
                    }
                    canvas?.drawPath(path, paint)
                }
            } catch (e: JSONException) {
                Log.w(TAG, "drawPolygon ${e}")

            }

        }
        fun drawLine(view: DrawCanvasView, canvas: Canvas?, p1: JSONObject?, p2: JSONObject?, paint: Paint) {
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

        fun drawPoints(
            view: DrawCanvasView,
            canvas: Canvas?,
            points: JSONArray,
            paint: Paint
        ) {
            try {
                repeat(points.length()) {
                    val point = view.getScreenCoordinate(points.getJSONObject(it))
                    canvas?.drawCircle(point[0], point[1], 10.0F, paint)

                    canvas?.drawText("${it + 1}" , point[0] + 15, point[1] + 15, paint)
                }
            } catch (e: JSONException) {
                Log.w(TAG, "drawPoints ${e}")

            }

        }

        fun drawPoint(view: DrawCanvasView, canvas: Canvas?, p: JSONObject, paint: Paint) {
            val point = view.getScreenCoordinate(p)
            canvas?.drawCircle(point[0], point[1], 2.0F, paint)
        }

        fun drawText(view: DrawCanvasView, canvas: Canvas?, text: String, p: JSONObject, paint: Paint) {
            val point = view.getScreenCoordinate(p)
            canvas?.drawText(text, point[0], point[1], paint)
        }


        fun drawPointKnot(view: DrawCanvasView, canvas: Canvas?, knots: JSONArray, paint: Paint){
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

        fun getAreaPathText(path: JSONArray, deltaX: Float, deltaY: Float, nColumn: Float, nRow: Float): String {
            val area = getAreaPath(path, deltaX, deltaY, nColumn, nRow)
            return " %.1f cm2".format(area / 100.0F)
        }

        fun getLengthPointText(p1: JSONObject, p2: JSONObject, deltaX: Float, deltaY: Float, nColumn: Float, nRow: Float): String {
            val l= getLengthPoint(p1, p2, deltaX, deltaY, nColumn, nRow)
            return " %.1f mm".format(l)
        }

    }

    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {

    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {

    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {

        when (annotationViewEvent) {
            is AnnotationViewEvent.RenderDraw -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(RenderDrawReducer(annotationActVM, it, annotationViewEvent ))

                }
            }

            is AnnotationViewEvent.ToggleAutoDraw -> {
                annotationActVM.setAutoDraw(annotationViewEvent.isAutoDraw)
            }

            is AnnotationViewEvent.ToggleManualDraw -> {
                annotationActVM.setManualDraw(annotationViewEvent.isManualDraw)
            }
        }
    }






    inner class RenderDrawReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.RenderDraw)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {


        override fun reduce(): AnnotationStateEffectObject {

            if (viewEvent.enableManualDraw) drawByType(viewState, viewModel, viewEvent, TYPE_DRAW_MANUAL)
            if (viewEvent.enableAutoDraw) drawByType(viewState, viewModel, viewEvent, TYPE_DRAW_AUTO)

            return AnnotationStateEffectObject()
        }

    }


    fun drawByType(viewState: AnnotationViewState, viewModel: AnnotationActVM, viewEvent: AnnotationViewEvent.RenderDraw, typeDraw: String) {

        val numFrame = viewModel.numFrame
        val currentFrameIndex = viewModel.getCurrentFrameIndex()
        val tool = viewModel.getCurrentTool()
        val currentToolId = tool.first
        val toolClickedType = tool.second
        val isGls = viewModel.getIsGls()
        val view = viewEvent.view
        val canvas = viewEvent.canvas
        val annotation = if (typeDraw == TYPE_DRAW_MANUAL) viewState.dicomAnnotation else viewState.machineAnnotation

        view.bitmap?.let {
            if (numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame) {

                val efPoints = annotation.getPointArray(frameIdx = currentFrameIndex, key = DicomAnnotation.EF_POINT)

                drawPoints(
                    view,
                    canvas,
                    efPoints,
                    paint = getPaintDrawPoint(typePoint = TYPE_POINT_EF, typeDraw = typeDraw)
                )

                val efBoundary = annotation.getBoundaryArray(frameIdx = currentFrameIndex, key = DicomAnnotation.EF_BOUNDARY)

                drawBoundary(
                    view,
                    canvas,
                    efBoundary,
                    paint = getPaintDrawLine(typeBoundary = TYPE_BOUNDARY_EF, typeDraw = typeDraw)
                )
//
                drawPolygon(
                    view,
                    canvas,
                    efBoundary,
                    paint = getPaintDrawPolygon(typeBoundary = TYPE_BOUNDARY_EF, typeDraw = typeDraw)
                )

                if (currentToolId == R.id.bt_draw_boundary && toolClickedType == TAG_LONG_CLICKED && isGls == false)
                    drawPointKnot(
                        view,
                        canvas,
                        knots = annotation.getKnots(currentFrameIndex, key = DicomAnnotation.EF_BOUNDARY),
                        paint = getPaintDrawLine(typeBoundary = TYPE_BOUNDARY_GLS, typeDraw = typeDraw)
                    )

                val glsPoints = annotation.getPointArray(frameIdx = currentFrameIndex, key = DicomAnnotation.GLS_POINT)
                drawPoints(
                    view,
                    canvas,
                    glsPoints,
                    getPaintDrawPoint(typePoint = TYPE_POINT_GLS, typeDraw = typeDraw)
                )

                val glsBoundary = annotation.getBoundaryArray(frameIdx = currentFrameIndex, key = DicomAnnotation.GLS_BOUNDARY)
                drawBoundary(
                    view,
                    canvas,
                    glsBoundary,
                    paint = getPaintDrawLine(typeBoundary = TYPE_BOUNDARY_GLS, typeDraw = typeDraw)
                )
//
                drawPolygon(
                    view,
                    canvas,
                    glsBoundary,
                    paint = getPaintDrawPolygon(typeBoundary = TYPE_BOUNDARY_GLS, typeDraw = typeDraw)
                )

                if (currentToolId == R.id.bt_draw_boundary && toolClickedType == TAG_LONG_CLICKED && isGls == true)
                    drawPointKnot(
                        view,
                        canvas,
                        knots = annotation.getKnots(currentFrameIndex, key = DicomAnnotation.GLS_BOUNDARY),
                        paint = getPaintDrawLine(typeBoundary = TYPE_BOUNDARY_GLS, typeDraw = typeDraw)
                    )


                val pointLengthTool = annotation.getPointArray(frameIdx = currentFrameIndex, key = DicomAnnotation.MEASURE_LENGTH)
                drawPointLengthTool(
                    viewState,
                    view,
                    canvas,
                    pointLengthTool,
                    paintText = getPaintDrawText(),
                    paintLine = getPaintDrawLine(
                        typeBoundary = TYPE_BOUNDARY_MEASURE_AREA, typeDraw = typeDraw
                    ),
                    paintPoint = getPaintDrawPoint(typePoint = TYPE_POINT_MEASURE_LENGTH, typeDraw = typeDraw)
                )


                val boundaryAreaTool = annotation.getBoundaryArray(frameIdx = currentFrameIndex, key = DicomAnnotation.MEASURE_AREA)
                drawBoundaryAreaTool(
                    viewState,
                    view,
                    canvas,
                    boundaryAreaTool,
                    paintText = getPaintDrawText(),
                    paintLine = getPaintDrawLine(
                        typeBoundary = TYPE_BOUNDARY_MEASURE_AREA, typeDraw = typeDraw
                    ),
                    paintPoint = getPaintDrawPoint(typePoint = TYPE_POINT_MEASURE_LENGTH, typeDraw = typeDraw)
                )
            }
        }
    }

    private fun drawBoundaryAreaTool(viewState: AnnotationViewState, view: DrawCanvasView, canvas: Canvas?, boundary: JSONArray, paintText: Paint, paintLine: Paint, paintPoint: Paint) {
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

    private fun drawPointLengthTool(viewState: AnnotationViewState, view: DrawCanvasView, canvas: Canvas?, points: JSONArray, paintText: Paint, paintLine: Paint, paintPoint: Paint) {
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
}