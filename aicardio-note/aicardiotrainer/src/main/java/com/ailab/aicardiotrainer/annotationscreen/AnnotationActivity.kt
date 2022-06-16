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

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.view.DragEvent
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import androidx.activity.viewModels
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.recyclerview.widget.LinearSmoothScroller
import com.ailab.aicardiotrainer.repositories.DicomAnnotation
import com.ailab.aicardiotrainer.repositories.DicomDiagnosis
import com.ailab.aicardiotrainer.repositories.User
import com.ailab.aicardiotrainer.repositories.FrameItem
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.api.ProgressListener
import com.ailab.aicardiotrainer.interfaces.OnDrawListener
import com.ailab.aicardiotrainer.interfaces.OnNormalizeTouchListener
import com.ailab.aicardiotrainer.studyscreen.DicomItem
import com.ailab.aicardiotrainer.toast
import com.rohitss.aacmvi.AacMviActivity
import kotlinx.android.synthetic.main.activity_annotation.*
import kotlinx.android.synthetic.main.dialog_progress.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch


class AnnotationActivity
    :  AacMviActivity<AnnotationViewState, AnnotationViewEffect, AnnotationViewEvent, AnnotationActVM>(),
    ProgressListener, OnDrawListener,
    OnNormalizeTouchListener, View.OnDragListener, View.OnTouchListener,
    DiagnosisDialog.OnDiagnosisEnteredListener, SaveDialog.OnSaveConfirmedListener, LoginDialog.OnSaveLoginListener, CopyAllFramesDialog.OnSaveCopyAllFramesListener{

    val handler = Handler()
    // <id button, <button, <short_clicked_text, long_click_ed_text> > >

    var tools: HashMap<Int, Pair<Button, Pair<String, String> > > = HashMap<Int, Pair<Button, Pair<String, String> > > ()

    companion object {

        val INTENT_FOLDER = "folder"
        val INTENT_FILE = "file"
        val TAG = "AnnotationActivity"
        val frameInterval = 30
        var bitmapHeart : Bitmap? = null
        var bitmapPlay : Bitmap? = null
        var bitmapPause : Bitmap? = null

        const val INTENT_KEY_DICOM_JPG = "DICOM_PATH_JPG"

        fun createIntent(context: Context, item: DicomItem): Intent {
            val intent = Intent(context, AnnotationActivity::class.java)
            intent.putExtra(INTENT_KEY_DICOM_JPG, item.imgPath)
            return intent
        }

    }

    override val viewModel: AnnotationActVM by viewModels()

    override fun onStop() {
        super.onStop()
        handler.removeCallbacksAndMessages(null)
    }

    private fun pushVideoToCanvas(handler: Handler) {
        handler.postDelayed({
            PlaybackMVI.process(viewModel, AnnotationViewEvent.NextFrame)
            pushVideoToCanvas(handler)
        }, frameInterval.toLong())
    }

    override fun onStart() {
        super.onStart()
        Log.w(TAG, "pushVideoToCanvas")

        pushVideoToCanvas(handler)
//
//        newsRvFolderAdapter.submitList(viewModel.getArrayFolderItem())
//
//        newsRvFolderAdapter.setCurrentPosition(viewModel.getFileName())
//
        newsRvFrameAdapter.submitList(viewModel.getListFrameList())
//
        newsRvFrameAdapter.setCurrentPosition(viewModel.getCurrentFrameIndex())
//
//        val toolUsed = viewModel.getCurrentTool()
//        val currentToolId = toolUsed.first
//        val typeClicked = toolUsed.second
//
//        // TODO Check null value
//        if (currentToolId != null && typeClicked != null) {
//            SetToolUsingMVI.renderViewEffect(annotationActivity = this, viewEffect = AnnotationViewEffect.RenderButtonTool(currentToolId, typeClicked))
//        }
//
//
//        val diagnosis = viewModel.getDiagnosis()
////        Log.w("TAG diagnosis", "$diagnosis")
//
//        SetToolUsingMVI.renderViewEffect(this, AnnotationViewEffect.RenderDiagnosisTool(diagnosis))
//
//        PlaybackMVI.renderViewEffect(annotationActivity = this, viewEffect = AnnotationViewEffect.RenderButtonPlayPause(button = R.id.bt_play_pause, isPlaying=viewModel.getIsPlaying()))
//
//        val phone = com.ailab.aicardio.repository.User.getPhone(this)
//
//        phone?.let {
//            SaveDataMVI.renderViewEffect(this, AnnotationViewEffect.RenderLoginButton(phone))
//        }?:run {
//            SaveDataMVI.renderViewEffect(this, AnnotationViewEffect.RenderLoginButton(com.ailab.aicardio.repository.User.DEFAULT_PHONE))
//        }
//
//        viewModel.getRenderAnnotationFrame()?.let {
//            PlaybackMVI.renderViewEffect(this, AnnotationViewEffect.RenderAnnotationFrame(it.renderAnnotation))
//        }

    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        iv_draw_canvas.saveToBundle(outState)
    }

    override fun onRestoreInstanceState(savedInstanceState: Bundle) {

        super.onRestoreInstanceState(savedInstanceState)
        iv_draw_canvas.getFromBundle(savedInstanceState)

    }

    val newsRvFrameAdapter by lazy {
        FrameRvAdapter(listener = {
            viewModel.process(
                AnnotationViewEvent.NewsFrameClicked(
                    it.tag as FrameItem
                )
            )
        }, longListener = {

            viewModel.process(AnnotationViewEvent.NewsFrameLongClicked(it.tag as FrameItem))
//            Logger.getLogger("TAG").warning("Long Clicked")
            true
        }, isVertical = true, annotationActVM = viewModel)
    }


    val downloadDicomFileMVI = DownloadDicomFileMVI(this)


    override fun onCreate(savedInstanceState: Bundle?) {


        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_annotation)

        iv_draw_canvas.setOnNormalizeTouchListener(this)
        iv_draw_canvas.setOnDrawListener(this)

        tools.put(R.id.bt_draw_point, Pair(bt_draw_point, Pair("POINT", "M-POINT") ) )
        tools.put(R.id.bt_draw_boundary, Pair(bt_draw_boundary, Pair("BOUNDARY", "M-BOUNDARY") ))

        tools.put(R.id.bt_diagnosis, Pair(bt_diagnosis, Pair("LABEL", "M-LABEL") ))

        tools.put(R.id.bt_measure_length, Pair(bt_measure_length, Pair("L", "M-L") ))
        tools.put(R.id.bt_measure_area, Pair(bt_measure_area, Pair("A", "M-A") ))

        rv_frame_info_list.adapter = newsRvFrameAdapter

        // full screen

//        bt_draw_full_screen.setOnClickListener {
//            bt_rv_folder_list.visibility = when (bt_rv_folder_list.visibility) {
//                View.VISIBLE -> View.GONE
//                else -> View.VISIBLE
//            }
//        }

        // login
//        bt_login.setOnClickListener {
//            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnUserLogin)
//        }
//
//        // NOTE: save data to disk when click (next, prev, play/ pause button)
        bt_play_pause.setOnClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.PlayPauseVideo)
        }

        bt_next_frame.setOnClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowNextFrame)
//            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveDataToDisk())

        }
        bt_prev_frame.setOnClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowPreviousFrame)
//            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveDataToDisk())

        }

        bt_next_frame.setOnLongClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowLastFrame)
            true
        }

        bt_prev_frame.setOnLongClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowFirstFrame)
            true
        }
//
//        bt_undo.setOnClickListener {
//            ProcessUndoAndClearMVI.process(viewModel, AnnotationViewEvent.UndoAnnotation)
//        }
//
//        bt_clear.setOnClickListener {
//            ProcessUndoAndClearMVI.process(viewModel, AnnotationViewEvent.ClearAnnotation)
//        }
//
//        bt_auto_analysis.setOnClickListener {
//            AutoAnalysisMVI.process(viewModel, AnnotationViewEvent.AutoAnalysis)
//        }

        // tools

//        bt_zooming_and_drag.setOnClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.bt_zooming_and_drag, typeClicked= TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
//        }
//
//        bt_draw_point.setOnClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.bt_draw_point, typeClicked= TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
//        }
//
//        bt_draw_point.setOnLongClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.bt_draw_point, typeClicked= TAG_LONG_CLICKED, isPlaying = viewModel.getIsPlaying()))
//            true
//        }
//
//        bt_draw_boundary.setOnClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.bt_draw_boundary, typeClicked= TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
//        }
//
//        bt_draw_boundary.setOnLongClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.bt_draw_boundary, typeClicked= TAG_LONG_CLICKED, isPlaying = viewModel.getIsPlaying()))
//            true
//        }
//
//        cb_perimeter.setOnClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.cb_perimeter, typeClicked=cb_perimeter.isChecked.toString(), isPlaying = viewModel.getIsPlaying()))
//        }
//
//        cb_draw_auto.setOnClickListener {
//            RenderDrawMVI.process(viewModel, AnnotationViewEvent.ToggleAutoDraw(cb_draw_auto.isChecked))
//            iv_draw_canvas.invalidate()
//        }
//
//        cb_draw_manual.setOnClickListener {
//            RenderDrawMVI.process(viewModel, AnnotationViewEvent.ToggleManualDraw(cb_draw_manual.isChecked))
//            iv_draw_canvas.invalidate()
//        }
//
//        cb_esv.setOnClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.cb_esv, typeClicked=cb_esv.isChecked.toString(), isPlaying = viewModel.getIsPlaying()))
//        }
//
//        cb_edv.setOnClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.cb_edv, typeClicked=cb_edv.isChecked.toString(), isPlaying = viewModel.getIsPlaying()))
//        }
//
//
//
//        bt_next_esv_edv_or_annotation.setOnClickListener {
//            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowEsvEdvOrAnnotationFrame(isEsvEdv = false))
//        }
//
//        bt_next_esv_edv_or_annotation.setOnLongClickListener {
//            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowEsvEdvOrAnnotationFrame(isEsvEdv = true))
//            true
//        }
//
//
//
//        // measurement tool
//        bt_measure_length.setOnClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.bt_measure_length, typeClicked= TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
//        }
//
//        bt_measure_area.setOnClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.bt_measure_area, typeClicked= TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
//        }


        // drag worksheet calculator
        layout_calculator.setOnTouchListener(this)
        layout_annotation.setOnDragListener(this)

        // TODO: when user click to button then set all checkbox value = false
        // Button Label - tqlong
//        bt_diagnosis.setOnClickListener {
//            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId= R.id.bt_diagnosis, typeClicked= TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
//        }
//
//        bt_save_data.setOnClickListener {
//            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveData)
//
//        }
//
//        bt_auto_copy.setOnClickListener {
//            CopyAutoAnnotationMVI.process(viewModel, AnnotationViewEvent.CopyAutoOneFrame)
//        }
//
//        bt_auto_copy.setOnLongClickListener {
//            showCopyAllFramesDialog()
//            true
//        }

    }






    override fun renderViewState(viewState: AnnotationViewState) {
        downloadDicomFileMVI.renderViewState(this, viewState)
        ReadDicomFileMVI.renderViewState(this, viewModel, viewState)
//        PlaybackMVI.renderViewState(this, viewState)

        when(viewState.status) {

            AnnotationViewStatus.Start -> {
                val dicomJPGPath = intent.getStringExtra(INTENT_KEY_DICOM_JPG)
                dicomJPGPath?.let {
                    downloadDicomFileMVI.process(viewModel, AnnotationViewEvent.DownloadDicom(dicomJPGPath))
                }
            }


        }
    }

    override fun renderViewEffect(viewEffect: AnnotationViewEffect) {
        ReadDicomFileMVI.renderViewEffect(this, viewEffect)
        PlaybackMVI.renderViewEffect(this, viewEffect)

        when(viewEffect) {
            is AnnotationViewEffect.ShowToast -> {
                toast(viewEffect.message)
            }
        }
    }

    override fun update(bytesRead: Long, contentLength: Long, done: Boolean) {
        Log.w(TAG, "Process to: ${bytesRead} ${contentLength} DONE: ${(100 * bytesRead) / contentLength} %")
        GlobalScope.launch(Dispatchers.Main) {
            if (done) closeProgressDialog()
            progressDialog?.let {
                it.tv_progress_percentage.text = ((100 * bytesRead) / contentLength).toString()
                it.pb_progress.progress = ((100 * bytesRead) / contentLength).toInt()
            }
        }
    }

    var progressDialog : ProgressDialog? = null

    fun openProgressDialog() {
        progressDialog = ProgressDialog(this)
        progressDialog!!.setCanceledOnTouchOutside(false)
        progressDialog!!.show()
    }

    fun closeProgressDialog() {
        progressDialog?.let {
            it.dismiss()
            progressDialog = null
        }
    }

    override fun onTouchEvent(view: DrawCanvasView, event: MotionEvent?, ix: Float, iy: Float) {
    }

    override fun onTouchEvent(view: com.ailab.aicardiotrainer.interpretation.InterpretationCanvasView, event: MotionEvent?, ix: Float, iy: Float) {
    }

    override fun draw(view: DrawCanvasView, canvas: Canvas?) {
    }

    override fun draw(view: com.ailab.aicardiotrainer.interpretation.InterpretationCanvasView, canvas: Canvas?) {
    }

    override fun onDrag(view: View?, dragEvent: DragEvent?): Boolean {
        when (dragEvent?.action) {
            DragEvent.ACTION_DRAG_ENDED, DragEvent.ACTION_DRAG_EXITED, DragEvent.ACTION_DRAG_ENTERED, DragEvent.ACTION_DRAG_STARTED, DragEvent.ACTION_DRAG_LOCATION -> {
                return true
            }
            DragEvent.ACTION_DROP -> {
                val tvState = dragEvent.localState as View
                val tvParent = tvState.parent as ViewGroup
                tvParent.removeView(tvState)
                val container = view as ConstraintLayout
                container.addView(tvState)
                tvParent.removeView(tvState)
                tvState.x = dragEvent.x - tvState.width/2
                tvState.y = dragEvent.y - tvState.height/2
                view.addView(tvState)
                view.setVisibility(View.VISIBLE)
                return true
            }

            else -> return false
        }
    }

    override fun onTouch(view: View?, motionEvent: MotionEvent?): Boolean {
        motionEvent?.let {
            return when (it.action) {
                MotionEvent.ACTION_DOWN -> {
                    val data = ClipData.newPlainText("", "")
                    val dragShadowBuilder = View.DragShadowBuilder(view)
                    view?.startDrag(data, dragShadowBuilder, view, 0)
                    true
                }
                else -> {
                    false
                }
            }
        } ?: run {
            return false
        }
    }

    override fun onSaveCopyAllFrames() {
    }

//    override fun onDiagnosisEntered(dicomDiagnosis: DicomDiagnosis) {
//        TODO("Not yet implemented")
//    }
//
//    override fun onSaveConfirmed(file: String, dicomAnnotation: DicomAnnotation, dicomDiagnosis: DicomDiagnosis) {
//        TODO("Not yet implemented")
//    }
//
//    override fun onSaveLogin(user: User) {
//        TODO("Not yet implemented")
//    }

    override fun onDiagnosisEntered(dicomDiagnosis: DicomDiagnosis) {
        Log.w(TAG, "onDiagnosisEntered")
    }

    override fun onSaveConfirmed(file: String, dicomAnnotation: DicomAnnotation, dicomDiagnosis: DicomDiagnosis) {
        Log.w(TAG, "onSaveConfirmed")
    }

    override fun onSaveLogin(user: User) {
        Log.w(TAG, "onSaveLogin")
    }
    class CenterSmoothScroller(context: Context?) : LinearSmoothScroller(context) {

        override fun calculateDtToFit(viewStart: Int, viewEnd: Int, boxStart: Int, boxEnd: Int, snapPreference: Int): Int {
            return boxStart + (boxEnd - boxStart) / 2 - (viewStart + (viewEnd - viewStart) / 2)
        }

    }


}
