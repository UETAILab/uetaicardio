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

package com.uetailab.aipacs.home_pacs.fragment_intepretation

import android.app.Activity
import android.app.Dialog
import android.content.Context
import android.graphics.Canvas
import android.graphics.Rect
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import com.uetailab.aipacs.R
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewDrawTouchEventMVI.Companion.drawBoundaryAreaTool
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewDrawTouchEventMVI.Companion.drawPointLengthTool
import kotlinx.android.synthetic.main.interpretation_view_dialog_ef_calculation.*
import kotlin.math.abs

class InterpretationViewEFCalculationDialog(val activity: Activity, val interpretationViewVM: InterpretationViewVM) : Dialog(activity) {

    companion object {
        const val TAG = "InterpretationViewEFCalculation"
    }
    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)

        val displayRectangle = Rect()
        val window: Window = activity.window
        window.decorView.getWindowVisibleDisplayFrame(displayRectangle)

        // inflate and adjust layout
        val inflater : LayoutInflater = activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val layout: View = inflater.inflate(R.layout.interpretation_view_dialog_ef_calculation, null)
//
        layout.minimumWidth = displayRectangle.width()
        layout.minimumHeight = displayRectangle.height()

        setContentView(layout)

        bt_cancel_dialog_ef_calculation.setOnClickListener {
            cancel()
        }

        bt_ok_dialog_ef_calculation.setOnClickListener {
            dismiss()
        }

        val efObject = interpretationViewVM.efObject

        val indexESV = efObject.indexESV
        val indexEDV = efObject.indexEDV

        val volumeESV =  53.49// efObject.volumeESV
        val volumeEDV =  132.71 // efObject.volumeEDV

        val efValue = 0.5969F // efObject.efValue
        val glsValue = -17.28 //interpretationViewVM.getGlsValue()


//        " %.1f cm2".format(result / 100.0F)
        if (indexEDV >= 0 && indexEDV >= 0) {
            val bitmap_esv = interpretationViewVM.getBitmapWithFrameID(indexESV)
            iv_draw_canvas_esv.setFitScale(bitmap_esv)
            iv_draw_canvas_esv.setCustomImageBitmap(bitmap_esv)

            val textESV = "Frame ES: ${indexESV + 1}/ ${interpretationViewVM.numFrame} Volume: ${"%.2f".format(volumeESV)}mL EF: ${"%.0f".format(efValue * 100)}% GLS: ${"%.2f".format(glsValue)}"
            iv_draw_canvas_esv.infoText = textESV

            iv_draw_canvas_esv.setOnDrawListener(object : OnDrawListener{
                override fun draw(view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?) {
                    interpretationViewVM.viewStates().value?.let {
                        drawCanvasEFCalculationDialogByType(it, interpretationViewVM, indexESV, view, canvas)
                    }
                }
            })


            val bitmap_edv = interpretationViewVM.getBitmapWithFrameID(indexEDV)
            iv_draw_canvas_edv.setFitScale(bitmap_edv)
            iv_draw_canvas_edv.setCustomImageBitmap(bitmap_edv)
            val textEDV = "Frame ED: ${indexEDV + 1}/ ${interpretationViewVM.numFrame} Volume: ${"%.2f".format(volumeEDV)}mL EF: ${"%.0f".format(efValue * 100)}% GLS: ${"%.2f".format(glsValue)}"

            iv_draw_canvas_edv.infoText = textEDV

            iv_draw_canvas_edv.setOnDrawListener(object : OnDrawListener{
                override fun draw(view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?) {
                    interpretationViewVM.viewStates().value?.let {
                        drawCanvasEFCalculationDialogByType(it, interpretationViewVM, indexEDV, view, canvas)
                    }
                }
            })
        }


//        val data_edv = interpretationViewVM.getMetaDataFrame(esv_edv_index.second)
//

    }

    fun drawCanvasEFCalculationDialogByType(viewState: InterpretationViewState, viewModel: InterpretationViewVM, currentFrameIndex: Int, view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?, typeDraw: String=InterpretationViewDrawTouchEventMVI.TYPE_DRAW_MANUAL) {

        val numFrame = viewModel.numFrame

        view.bitmap?.let {
            val annotation: DicomInterpretation = if (typeDraw == InterpretationViewDrawTouchEventMVI.TYPE_DRAW_MANUAL) viewState.dicomInterpretation else viewState.machineInterpretation
            if (numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame) {
                // draw for ef (point, boundary)
                val efPoints = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.EF_POINT)
                val efBoundary = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.EF_BOUNDARY)

                val length = InterpretationViewDrawTouchEventMVI.drawPointsDialog(
                    viewState,
                    view,
                    canvas,
                    efPoints,
                    paint = InterpretationViewDrawTouchEventMVI.getPaintDrawPoint(
                        typePoint = InterpretationViewDrawTouchEventMVI.TYPE_POINT_EF,
                        typeDraw = typeDraw
                    )
                )
                val area = InterpretationViewDrawTouchEventMVI.drawBoundary(
                    viewState,
                    view,
                    canvas,
                    efBoundary,
                    paint = InterpretationViewDrawTouchEventMVI.getPaintDrawLine(
                        typeBoundary = InterpretationViewDrawTouchEventMVI.TYPE_BOUNDARY_EF,
                        typeDraw = typeDraw
                    )
                )
                InterpretationViewDrawTouchEventMVI.drawMultiPolygon(
                    viewState,
                    view,
                    canvas,
                    efBoundary,
                    paint = InterpretationViewDrawTouchEventMVI.getPaintDrawPolygon(
                        typeBoundary = InterpretationViewDrawTouchEventMVI.TYPE_BOUNDARY_EF,
                        typeDraw = typeDraw
                    )
                )

                val volume = if (abs(length - 0F) < DicomInterpretation.EPS) 0F else (0.85F * area * area) / length / 1000.0F // mL
                if (volume > 0) {
                    // draw volume at index 2
                    val point = view.getScreenCoordinate(efPoints.getJSONObject(6))
                    canvas?.drawText(" %.2f mL".format(volume) , point[0] + 30, point[1] + 30, InterpretationViewDrawTouchEventMVI.getPaintDrawText())

                }


                // draw for gls (point, boundary)
                val glsPoints = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.GLS_POINT)
                val glsBoundary = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.GLS_BOUNDARY)

                InterpretationViewDrawTouchEventMVI.drawPoints(
                    viewState,
                    view,
                    canvas,
                    glsPoints,
                    InterpretationViewDrawTouchEventMVI.getPaintDrawPoint(
                        typePoint = InterpretationViewDrawTouchEventMVI.TYPE_POINT_GLS,
                        typeDraw = typeDraw
                    )
                )
                InterpretationViewDrawTouchEventMVI.drawBoundary(
                    viewState,
                    view,
                    canvas,
                    glsBoundary,
                    paint = InterpretationViewDrawTouchEventMVI.getPaintDrawLine(
                        typeBoundary = InterpretationViewDrawTouchEventMVI.TYPE_BOUNDARY_GLS,
                        typeDraw = typeDraw
                    )
                )
                InterpretationViewDrawTouchEventMVI.drawMultiPolygon(
                    viewState,
                    view,
                    canvas,
                    glsBoundary,
                    paint = InterpretationViewDrawTouchEventMVI.getPaintDrawPolygon(
                        typeBoundary = InterpretationViewDrawTouchEventMVI.TYPE_BOUNDARY_GLS,
                        typeDraw = typeDraw
                    )
                )


//                // mod draw length
                val pointLengthTool = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.MEASURE_LENGTH)
//
                drawPointLengthTool( viewState, view, canvas, pointLengthTool, paintText = InterpretationViewDrawTouchEventMVI.getPaintDrawText(), paintLine = InterpretationViewDrawTouchEventMVI.getPaintDrawLine(
                    typeBoundary = InterpretationViewDrawTouchEventMVI.TYPE_BOUNDARY_MEASURE_AREA,
                    typeDraw = typeDraw
                ), paintPoint = InterpretationViewDrawTouchEventMVI.getPaintDrawPoint(
                    typePoint = InterpretationViewDrawTouchEventMVI.TYPE_POINT_MEASURE_LENGTH,
                    typeDraw = typeDraw))

//                // mod draw area
                val boundaryAreaTool = annotation.getFramePointArrayWithKey(frameIdx = currentFrameIndex, key = FrameAnnotation.MEASURE_AREA)
                drawBoundaryAreaTool( viewState, view, canvas, boundaryAreaTool, paintText = InterpretationViewDrawTouchEventMVI.getPaintDrawText(),
                    paintLine = InterpretationViewDrawTouchEventMVI.getPaintDrawLine(
                    typeBoundary = InterpretationViewDrawTouchEventMVI.TYPE_BOUNDARY_MEASURE_AREA,
                    typeDraw = typeDraw
                ), paintPoint = InterpretationViewDrawTouchEventMVI.getPaintDrawPoint(
                    typePoint = InterpretationViewDrawTouchEventMVI.TYPE_POINT_MEASURE_LENGTH,
                    typeDraw = typeDraw
                ))
//
            }
        }
    }

}