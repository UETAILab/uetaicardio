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

package com.ailab.aicardiotrainer.studyscreen

import android.app.Activity
import android.app.Dialog
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Rect
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import com.ailab.aicardiotrainer.R
import kotlinx.android.synthetic.main.dialog_dicom_preview.*


class DicomPreviewDialog(
    val activity: StudyActivity,
//    val listener: OnSaveConfirmedListener,
    val file: String,
    val bitmaps: List<Bitmap>
) : Dialog(activity),
    OnDrawListener {
    companion object {
        val TAG = "DicomPreviewDialog"
        val frameInterval: Int = 30
    }

    var currentFrameIndex : Int = 0
    val numFrame : Int get() = bitmaps.size
    val handler = Handler()
    var isPlaying: Boolean = true
    interface OnSaveConfirmedListener {
//        fun onSaveConfirmed(file: String, dicomAnnotation: DicomAnnotation, dicomDiagnosis: DicomDiagnosis)
    }
//    val currentFrameInActivity = (activity as AnnotationActivity).viewModel.getCurrentFrameIndex()


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // retrieve display dimensions
        val displayRectangle = Rect()
        val window: Window = activity.window
        window.decorView.getWindowVisibleDisplayFrame(displayRectangle)

        // inflate and adjust layout
        val inflater : LayoutInflater = activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val layout: View = inflater.inflate(R.layout.dialog_dicom_preview, null)
        layout.minimumWidth = (displayRectangle.width() * 0.9f).toInt()
        layout.minimumHeight = (displayRectangle.height() * 0.9f).toInt()
        setContentView(layout)
//        requestWindowFeature(Window.FEATURE_NO_TITLE)
//        setContentView(R.layout.dialog_save_annotate)

        bt_cancel.setOnClickListener { cancel() }
        bt_copy.setOnClickListener { onSaveClicked() }

        bt_next_frame.setOnClickListener {
            Log.w(TAG, "bt_next_frame clicked ${currentFrameIndex}")

            showNextFrame()
        }

        bt_prev_frame.setOnClickListener {
            Log.w(TAG, "bt_prev_frame clicked ${currentFrameIndex}")
            showPreviousFrame()
        }

        bt_play_pause.setOnClickListener {
            if (activity.bitmapPlay == null) activity.bitmapPlay = BitmapFactory.decodeResource(activity.resources, R.drawable.ic_play)
            if (activity.bitmapPause == null) activity.bitmapPause = BitmapFactory.decodeResource(activity.resources, R.drawable.ic_pause)

            Log.w(TAG, "bt_play_pause clicked ${currentFrameIndex}")
            isPlaying = !isPlaying
            bt_play_pause.setImageBitmap(if (isPlaying) activity.bitmapPause else activity.bitmapPlay)
        }

//        iv_draw_canvas.setOnDrawListener(this)

        if (numFrame > 0) {
            iv_draw_canvas.setFitScale(bitmaps.get(0))
            iv_draw_canvas.setCustomImageBitmap(bitmaps.get(0))
        }
//        Log.w(TAG, "n_bitmap $numFrame")
    }

    override fun onStart() {
        super.onStart()
//        if (numFrame > 0 && isPlaying) pushVideoToCanvas()
        pushVideoToCanvas(handler)

    }

    override fun onStop() {
        super.onStop()
        handler.removeCallbacksAndMessages(null)
    }

    private fun pushVideoToCanvas(handler: Handler) {
        handler.postDelayed({
            if (numFrame > 0) {
//                currentFrameIndex = (currentFrameIndex + 1) % numFrame
                playNextFrame()
//                renderCurrentFrame()
            }
            pushVideoToCanvas(handler)
        }, frameInterval.toLong())
    }

    private fun onSaveClicked() {
//        listener.onSaveConfirmed(file, dicomAnnotation, dicomDiagnosis)
        dismiss()
    }



    override fun draw(view: DicomPreviewCanvasView, canvas: Canvas?) {
//        TODO("Not yet implemented")
    }


    private fun renderCurrentFrame() {
        if (numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame) {
            iv_draw_canvas.setCustomImageBitmap(bitmaps.get(currentFrameIndex))
            iv_draw_canvas.infoText = "${currentFrameIndex + 1} / ${numFrame}"
        }
    }

    private fun playNextFrame(){
        if (numFrame > 0 && isPlaying == true) {
            currentFrameIndex = (currentFrameIndex + 1) % numFrame
            renderCurrentFrame()
        }
    }

    private fun showFirstFrame(){
        if (numFrame > 0) {
            isPlaying = false
            currentFrameIndex = 0
            renderCurrentFrame()
        }
    }

    private fun showLastFrame() {
        if (numFrame > 0) {
            isPlaying = false

            currentFrameIndex = numFrame - 1
            renderCurrentFrame()
        }
    }

    private fun showNextFrame() {
        if (numFrame > 0) {
            isPlaying = false

            currentFrameIndex = (currentFrameIndex + 1) % numFrame
            renderCurrentFrame()
        }
    }

    private fun showPreviousFrame() {
        if (numFrame > 0) {
            isPlaying = false

            currentFrameIndex = currentFrameIndex - 1
            if (currentFrameIndex < 0) {
                currentFrameIndex = numFrame - 1
            }
            renderCurrentFrame()
        }
    }


}