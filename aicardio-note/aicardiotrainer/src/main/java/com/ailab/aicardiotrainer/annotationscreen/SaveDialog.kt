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

package com.ailab.aicardiotrainer.annotationscreen

import android.app.Activity
import android.app.Dialog
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import com.ailab.aicardiotrainer.*
import com.ailab.aicardiotrainer.interfaces.OnDrawListener
import com.ailab.aicardiotrainer.repositories.DicomAnnotation
import com.ailab.aicardiotrainer.repositories.DicomDiagnosis
import kotlinx.android.synthetic.main.dialog_save_annotate.*


class SaveDialog(
    val activity: Activity,
    val listener: OnSaveConfirmedListener,
    val file: String,
    val bitmaps: List<Bitmap>,
    val dicomAnnotation: DicomAnnotation,
    val dicomDiagnosis: DicomDiagnosis
) : Dialog(activity),
    OnDrawListener {
    companion object {
        val TAG = "SaveDialog"
        val frameInterval: Int = 30
    }

    var currentFrameIndex : Int = 0
    val numFrame : Int get() = bitmaps.size
    val handler = Handler()

    interface OnSaveConfirmedListener {
        fun onSaveConfirmed(file: String, dicomAnnotation: DicomAnnotation, dicomDiagnosis: DicomDiagnosis)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // retrieve display dimensions
        val displayRectangle = Rect()
        val window: Window = activity.window
        window.decorView.getWindowVisibleDisplayFrame(displayRectangle)

        // inflate and adjust layout
        val inflater : LayoutInflater = activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val layout: View = inflater.inflate(R.layout.dialog_save_annotate, null)
        layout.minimumWidth = (displayRectangle.width() * 0.9f).toInt()
        layout.minimumHeight = (displayRectangle.height() * 0.9f).toInt()
        setContentView(layout)
//        requestWindowFeature(Window.FEATURE_NO_TITLE)
//        setContentView(R.layout.dialog_save_annotate)

        bt_cancel.setOnClickListener { cancel() }
        bt_copy.setOnClickListener { onSaveClicked() }

        iv_draw_canvas.setOnDrawListener(this)

        if (numFrame > 0) {
            iv_draw_canvas.setFitScale(bitmaps.get(0))
            iv_draw_canvas.setCustomImageBitmap(bitmaps.get(0))
        }
        Log.w(TAG, "n_bitmap $numFrame")
    }

    override fun onStart() {
        super.onStart()
        if (numFrame > 0) pushVideoToCanvas()
    }

    override fun onStop() {
        super.onStop()
        handler.removeCallbacksAndMessages(null)
    }

    private fun pushVideoToCanvas() {
        handler.postDelayed({
            if (numFrame > 0) {
                currentFrameIndex = (currentFrameIndex + 1) % numFrame
                iv_draw_canvas.setCustomImageBitmap(bitmaps.get(currentFrameIndex))
            }
            pushVideoToCanvas()
        }, frameInterval.toLong())
    }

    private fun onSaveClicked() {
        listener.onSaveConfirmed(file, dicomAnnotation, dicomDiagnosis)
        dismiss()
    }

    override fun draw(view: DrawCanvasView, canvas: Canvas?) {

//        if (numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame) {
//            // DRAW POINT
////            viewState.efPoints
//            view.bitmap?.let {
//
//                val efPoints = dicomAnnotation.getPointArray(frameIdx = currentFrameIndex, key = DicomAnnotation.EF_POINT)
//                RenderDrawMVI.drawPoints(view, canvas, efPoints, paint= RenderDrawMVI.getPaintDrawPoint(
//                    typePoint = RenderDrawMVI.TYPE_POINT_EF, typeDraw = RenderDrawMVI.TYPE_DRAW_MANUAL
//                )
//                )
//
//                val efBoundary = dicomAnnotation.getBoundaryArray(frameIdx = currentFrameIndex, key = DicomAnnotation.EF_BOUNDARY)
//                RenderDrawMVI.drawBoundary(view, canvas, efBoundary, paint= RenderDrawMVI.getPaintDrawLine(
//                    typeBoundary = RenderDrawMVI.TYPE_BOUNDARY_EF, typeDraw = RenderDrawMVI.TYPE_DRAW_MANUAL
//                )
//                )
//                RenderDrawMVI.drawPolygon(view, canvas, efBoundary, paint= RenderDrawMVI.getPaintDrawPolygon(
//                    typeBoundary = RenderDrawMVI.TYPE_BOUNDARY_GLS, typeDraw = RenderDrawMVI.TYPE_DRAW_MANUAL
//                )
//                )
//
//                val glsPoints = dicomAnnotation.getPointArray(frameIdx = currentFrameIndex, key = DicomAnnotation.GLS_POINT)
//                RenderDrawMVI.drawPoints(view, canvas, glsPoints,
//                    RenderDrawMVI.getPaintDrawPoint(
//                        typePoint = RenderDrawMVI.TYPE_POINT_GLS, typeDraw = RenderDrawMVI.TYPE_DRAW_MANUAL
//                    )
//                )
//
//                val glsBoundary = dicomAnnotation.getBoundaryArray(frameIdx = currentFrameIndex, key = DicomAnnotation.GLS_BOUNDARY)
//                RenderDrawMVI.drawBoundary(view, canvas, glsBoundary, paint= RenderDrawMVI.getPaintDrawLine(
//                    typeBoundary = RenderDrawMVI.TYPE_BOUNDARY_GLS, typeDraw = RenderDrawMVI.TYPE_DRAW_MANUAL
//                )
//                )
//                RenderDrawMVI.drawPolygon(view, canvas, glsBoundary, paint= RenderDrawMVI.getPaintDrawPolygon(
//                    typeBoundary = RenderDrawMVI.TYPE_BOUNDARY_GLS, typeDraw = RenderDrawMVI.TYPE_DRAW_MANUAL
//                )
//                )
//
//            }
//        }
    }

    override fun draw(view: com.ailab.aicardiotrainer.interpretation.InterpretationCanvasView, canvas: Canvas?) {
        TODO("Not yet implemented")
    }
}