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

package com.uetailab.aipacs.home_pacs.fragment_home

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
import com.uetailab.aipacs.R
import com.uetailab.aipacs.home_pacs.HomePacsActivity
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewEFCalculationDialog
import kotlinx.android.synthetic.main.home_view_dialog_dicom_preview.*


class HomeViewStudyPreviewDialog(
    val activity: HomePacsActivity,
    val file: String,
    val bitmaps: List<Bitmap>
) : Dialog(activity),
    OnDrawListener {
    companion object {
        val TAG = "HomeViewStudyPreviewDialog"
        val frameInterval: Int = 30
        var bitmapPlay : Bitmap? = null
        var bitmapPause : Bitmap? = null

    }

    var currentFrameIndex : Int = 0
    val numFrame : Int get() = bitmaps.size
    val handler = Handler()
    var isPlaying: Boolean = true
    var shortNameFile = file.substringAfterLast("____")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // retrieve display dimensions
        val displayRectangle = Rect()
        val window: Window = activity.window
        window.decorView.getWindowVisibleDisplayFrame(displayRectangle)

        // inflate and adjust layout
        val inflater : LayoutInflater = activity.getSystemService(Context.LAYOUT_INFLATER_SERVICE) as LayoutInflater
        val layout: View = inflater.inflate(R.layout.home_view_dialog_dicom_preview, null)
        layout.minimumWidth = (displayRectangle.width() * 0.9f).toInt()
        layout.minimumHeight = (displayRectangle.height() * 0.9f).toInt()
        Log.w(TAG, "w: ${layout.minimumWidth} h: ${layout.minimumHeight}" )

        setContentView(layout)
//        requestWindowFeature(Window.FEATURE_NO_TITLE)
//        setContentView(R.layout.dialog_save_annotate)

        if (bitmapPlay == null) bitmapPlay = BitmapFactory.decodeResource(activity.resources, R.drawable.ic_play)
        if (bitmapPause == null) bitmapPause = BitmapFactory.decodeResource(activity.resources, R.drawable.ic_pause)

        bt_cancel_study_dialog_preview.setOnClickListener { cancel() }
        bt_ok_study_dialog_preview.setOnClickListener { onSaveClicked() }

        bt_dialog_next_frame.setOnClickListener {
            showNextFrame()
        }

        bt_dialog_prev_frame.setOnClickListener {
            showPreviousFrame()
        }

        bt_dialog_next_frame.setOnLongClickListener {
            showLastFrame()
            true
        }

        bt_dialog_prev_frame.setOnLongClickListener {
            showFirstFrame()
            true
        }

        bt_dialog_play_pause.setOnClickListener {
            if (numFrame > 1) {
                isPlaying = !isPlaying
            } else isPlaying = false
            bt_dialog_play_pause.setImageBitmap(if (isPlaying) bitmapPause else bitmapPlay)
        }

        if (numFrame == 1) {
            isPlaying = false
        }

        if (numFrame > 0) {
            iv_draw_canvas.setFitScale(bitmaps.get(0))
            iv_draw_canvas.setCustomImageBitmap(bitmaps.get(0))
            bt_dialog_play_pause.setImageBitmap(if (isPlaying) bitmapPause else bitmapPlay)
            iv_draw_canvas.infoText = " ${currentFrameIndex + 1} / ${numFrame} ${shortNameFile} "
        }

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
        dismiss()
    }

    override fun draw(view: HomeViewStudyPreviewCanvasView, canvas: Canvas?) {
//        TODO("Not yet implemented")
    }


    private fun renderCurrentFrame() {
        if (numFrame > 0 && currentFrameIndex >= 0 && currentFrameIndex < numFrame) {
            iv_draw_canvas.setCustomImageBitmap(bitmaps.get(currentFrameIndex))
            iv_draw_canvas.infoText = " ${currentFrameIndex + 1} / ${numFrame} ${shortNameFile} "
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