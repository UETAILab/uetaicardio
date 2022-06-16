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

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
import com.ailab.aicardio.R
import com.ailab.aicardio.TAG_LONG_CLICKED
import com.ailab.aicardio.TAG_SHORT_CLICKED
import com.ailab.aicardio.annotationscreen.interfaces.OnDrawListener
import com.ailab.aicardio.annotationscreen.interfaces.OnNormalizeTouchListener
import com.ailab.aicardio.annotationscreen.views.*
import com.ailab.aicardio.mainscreen.FolderRvAdapter
import com.ailab.aicardio.repository.*
import com.ailab.aicardio.toast
import com.rohitss.aacmvi.AacMviActivity
import kotlinx.android.synthetic.main.activity_annotate.*
import android.graphics.Point


@Suppress("DEPRECATION")
class AnnotationActivity : AacMviActivity<AnnotationViewState, AnnotationViewEffect, AnnotationViewEvent, AnnotationActVM>(),
    OnDrawListener,
    OnNormalizeTouchListener, View.OnDragListener, View.OnTouchListener,
    DiagnosisDialog.OnDiagnosisEnteredListener, SaveDialog.OnSaveConfirmedListener, LoginDialog.OnSaveLoginListener, CopyAllFramesDialog.OnSaveCopyAllFramesListener  {

    override val viewModel: AnnotationActVM by viewModels()

    val handler = Handler()
    // <id button, <button, <short_clicked_text, long_click_ed_text> > >

    var tools: HashMap<Int, Pair< Button, Pair<String, String> > > = HashMap<Int, Pair< Button, Pair<String, String> > > ()

    companion object {
        val INTENT_FOLDER = "folder"
        val INTENT_FILE = "file"
        val TAG = "AnnotationActivity"
        val frameInterval = 30
        var bitmapHeart : Bitmap? = null
        var bitmapPlay : Bitmap? = null
        var bitmapPause : Bitmap? = null

        fun createIntent(context: Context, folder: String, file: String?): Intent {
            val intent = Intent(context, AnnotationActivity::class.java)
            intent.putExtra(INTENT_FOLDER, folder)
            intent.putExtra(INTENT_FILE, file)
            return intent
        }

    }

    override fun onStop() {
        super.onStop()
        handler.removeCallbacksAndMessages(null)
    }

//    val phone: String = User.getPhone(this) User.DEFAULT_PHONE

    override fun onStart() {
        super.onStart()
        Log.w(TAG, "pushVideoToCanvas")
        pushVideoToCanvas(handler)

        newsRvFolderAdapter.submitList(viewModel.getArrayFolderItem())

        newsRvFolderAdapter.setCurrentPosition(viewModel.getFileName())

        newsRvFrameAdapter.submitList(viewModel.getListFrameList())

        newsRvFrameAdapter.setCurrentPosition(viewModel.getCurrentFrameIndex())

        val toolUsed = viewModel.getCurrentTool()
        val currentToolId = toolUsed.first
        val typeClicked = toolUsed.second

        // TODO Check null value
        if (currentToolId != null && typeClicked != null) {
            SetToolUsingMVI.renderViewEffect(annotationActivity = this, viewEffect = AnnotationViewEffect.RenderButtonTool(currentToolId, typeClicked))
        }


        val diagnosis = viewModel.getDiagnosis()
//        Log.w("TAG diagnosis", "$diagnosis")

        SetToolUsingMVI.renderViewEffect(this, AnnotationViewEffect.RenderDiagnosisTool(diagnosis))

        PlaybackMVI.renderViewEffect(annotationActivity = this, viewEffect = AnnotationViewEffect.RenderButtonPlayPause(button = R.id.bt_play_pause, isPlaying=viewModel.getIsPlaying()))

        val phone = User.getPhone(this)

        phone?.let {
            if (phone == User.DEFAULT_PHONE) {
                showUserLoginDialog()
            }
            SaveDataMVI.renderViewEffect(this, AnnotationViewEffect.RenderLoginButton(phone))
        }?:run {
            SaveDataMVI.renderViewEffect(this, AnnotationViewEffect.RenderLoginButton(User.DEFAULT_PHONE))
        }

        viewModel.getRenderAnnotationFrame()?.let {
            PlaybackMVI.renderViewEffect(this, AnnotationViewEffect.RenderAnnotationFrame(it.renderAnnotation))
        }

        Log.w(TAG, "${viewModel.viewStates().value?.boundaryHeart}")

//        viewModel.viewStates().value?.let {
//            it = it.copy()
//        }?: run {
//
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

    val newsRvFolderAdapter by lazy {
        FolderRvAdapter(listener =  {
            viewModel.process(
                AnnotationViewEvent.NewsItemFileClicked(
                    it.tag as FolderItem
                )
            )
        }, longListener = {

            viewModel.process(AnnotationViewEvent.NewsItemFileLongClicked(it.tag as FolderItem))

//            Logger.getLogger("TAG").warning("Long Clicked")
            true
        }, isVertical = true)
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



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_annotate)

        iv_draw_canvas.setOnNormalizeTouchListener(this)
        iv_draw_canvas.setOnDrawListener(this)


        tools.put(R.id.bt_draw_point, Pair(bt_draw_point, Pair("POINT", "M-POINT") ) )
        tools.put(R.id.bt_draw_boundary, Pair(bt_draw_boundary, Pair("BOUNDARY", "M-BOUNDARY") ))

        tools.put(R.id.bt_diagnosis, Pair(bt_diagnosis, Pair("LABEL", "M-LABEL") ))

        tools.put(R.id.bt_measure_length, Pair(bt_measure_length, Pair("L", "M-L") ))
        tools.put(R.id.bt_measure_area, Pair(bt_measure_area, Pair("A", "M-A") ))

        // full screen

        bt_draw_full_screen.setOnClickListener {
            bt_rv_folder_list.visibility = when (bt_rv_folder_list.visibility) {
                View.VISIBLE -> View.GONE
                else -> View.VISIBLE
            }
        }

        // login
        bt_login.setOnClickListener {
            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnUserLogin)
        }

        // rv_folder_list
        rv_folder_list.adapter = newsRvFolderAdapter

        rv_frame_info_list.adapter = newsRvFrameAdapter
        // NOTE: save data to disk when click (next, prev, play/ pause button)
        bt_play_pause.setOnClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.PlayPauseVideo)
            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveDataToDisk())

        }

        bt_next_frame.setOnClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowNextFrame)
            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveDataToDisk())

        }
        bt_prev_frame.setOnClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowPreviousFrame)
            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveDataToDisk())

        }

        bt_next_frame.setOnLongClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowLastFrame)
            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveDataToDisk())
            true
        }

        bt_prev_frame.setOnLongClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowFirstFrame)
            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveDataToDisk())
            true
        }

        bt_undo.setOnClickListener {
            ProcessUndoAndClearMVI.process(viewModel, AnnotationViewEvent.UndoAnnotation)
        }

        bt_clear.setOnClickListener {
            ProcessUndoAndClearMVI.process(viewModel, AnnotationViewEvent.ClearAnnotation)
        }

        bt_auto_analysis.setOnClickListener {
            AutoAnalysisMVI.process(viewModel, AnnotationViewEvent.AutoAnalysis)
        }

        // tools

        bt_zooming_and_drag.setOnClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.bt_zooming_and_drag, typeClicked=TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
        }

        bt_draw_point.setOnClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.bt_draw_point, typeClicked=TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
        }

        bt_draw_point.setOnLongClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.bt_draw_point, typeClicked= TAG_LONG_CLICKED, isPlaying = viewModel.getIsPlaying()))
            true
        }

        bt_draw_boundary.setOnClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.bt_draw_boundary, typeClicked=TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
        }

        bt_draw_boundary.setOnLongClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.bt_draw_boundary, typeClicked= TAG_LONG_CLICKED, isPlaying = viewModel.getIsPlaying()))
            true
        }

        cb_perimeter.setOnClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.cb_perimeter, typeClicked=cb_perimeter.isChecked.toString(), isPlaying = viewModel.getIsPlaying()))
        }

        cb_draw_auto.setOnClickListener {
            RenderDrawMVI.process(viewModel, AnnotationViewEvent.ToggleAutoDraw(cb_draw_auto.isChecked))
            iv_draw_canvas.invalidate()
        }

        cb_draw_manual.setOnClickListener {
            RenderDrawMVI.process(viewModel, AnnotationViewEvent.ToggleManualDraw(cb_draw_manual.isChecked))
            iv_draw_canvas.invalidate()
        }

        cb_esv.setOnClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.cb_esv, typeClicked=cb_esv.isChecked.toString(), isPlaying = viewModel.getIsPlaying()))
        }

        cb_edv.setOnClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.cb_edv, typeClicked=cb_edv.isChecked.toString(), isPlaying = viewModel.getIsPlaying()))
        }



        bt_next_esv_edv_or_annotation.setOnClickListener {
            Log.w(TAG, "Clicked bt_next_esv_edv_or_annotation")
            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowEsvEdvOrAnnotationFrame(isEsvEdv = false))
            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveDataToDisk())
        }

        bt_next_esv_edv_or_annotation.setOnLongClickListener {
            PlaybackMVI.process(viewModel, AnnotationViewEvent.ShowEsvEdvOrAnnotationFrame(isEsvEdv = true))
            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveDataToDisk())
            true
        }



        // measurement tool
        bt_measure_length.setOnClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.bt_measure_length, typeClicked=TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
        }

        bt_measure_area.setOnClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.bt_measure_area, typeClicked=TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
        }


        // drag worksheet calculator
        layout_calculator.setOnTouchListener(this)
        layout_annotation.setOnDragListener(this)

        // TODO: when user click to button then set all checkbox value = false
        // Button Label - tqlong
        bt_diagnosis.setOnClickListener {
            SetToolUsingMVI.process(viewModel, AnnotationViewEvent.OnToolUsing(toolId=R.id.bt_diagnosis, typeClicked=TAG_SHORT_CLICKED, isPlaying = viewModel.getIsPlaying()))
        }

        bt_save_data.setOnClickListener {
            SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveData)

        }

        bt_auto_copy.setOnClickListener {
            CopyAutoAnnotationMVI.process(viewModel, AnnotationViewEvent.CopyAutoOneFrame)
        }

        bt_auto_copy.setOnLongClickListener {
            showCopyAllFramesDialog()
            true
        }

    }

    fun showDiagnosisDialog(dicomDiagnosis: DicomDiagnosis) {
        val dlg = DiagnosisDialog(
            this,
            this,
            dicomDiagnosis,
            viewModel
        )
        dlg.setCanceledOnTouchOutside(false)
        dlg.show()
    }


    private fun showCopyAllFramesDialog() {
        val dlg = CopyAllFramesDialog(
            this,
            this
        )
        dlg.setCanceledOnTouchOutside(false)
        dlg.show()
    }


    private fun pushVideoToCanvas(handler: Handler) {
        handler.postDelayed({
            PlaybackMVI.process(viewModel, AnnotationViewEvent.NextFrame)
            pushVideoToCanvas(handler)
        }, frameInterval.toLong())
    }

    override fun onTouchEvent(view: DrawCanvasView, event: MotionEvent?, ix: Float, iy: Float) {
        ProcessTouchEventMVI.process(viewModel, AnnotationViewEvent.ProcessTouchEvent(view, event, ix, iy))
    }

    override fun draw(view: DrawCanvasView, canvas: Canvas?) {
        RenderDrawMVI.process(viewModel, AnnotationViewEvent.RenderDraw(view, canvas, enableManualDraw = viewModel.getEnableManualDraw(), enableAutoDraw = viewModel.getEnableAutoDraw()))
    }

    override fun renderViewState(viewState: AnnotationViewState) {
//        Log.w(TAG, "Status: ${viewState.annotateViewStatus.javaClass}")
        if (bitmapHeart == null) bitmapHeart = BitmapFactory.decodeResource(this.resources, R.drawable.heart)

        UniDirectionMVI.renderViewState(this, viewState)

        AutoAnalysisMVI.renderViewState(this, viewState)

        CopyAutoAnnotationMVI.renderViewState(this, viewState)

        FetchNewsFolderMVI.renderViewState(this, viewState)

        FetchNewsFileMVI.renderViewState(this, viewModel, viewState)
        SaveDataMVI.renderViewState(this, viewState)

        when(viewState.status) {

            AnnotationViewStatus.NotFetched -> {
                Log.w(TAG, "not fetched")
                val folder = intent.getStringExtra(INTENT_FOLDER)
                val file = intent.getStringExtra(INTENT_FILE)
                iv_draw_canvas.setCustomImageBitmap(bitmapHeart)
                FetchNewsBothFolderAndFileMVI.process(viewModel, AnnotationViewEvent.FetchNewsBothFolderAndFile(folder=folder, file=file))

            }


            is AnnotationViewStatus.OpenAnnotationActivity -> {
                val folder = viewState.status.folder
                if (folder.contains(FolderRepository.DEFAULT_FOLDER_DOWNLOAD)) {
                    val intent = createIntent(applicationContext, folder = folder, file = null)
                    startActivity(intent)
                }
            }
        }

    }

    class CenterSmoothScroller(context: Context?) : LinearSmoothScroller(context) {

        override fun calculateDtToFit(viewStart: Int, viewEnd: Int, boxStart: Int, boxEnd: Int, snapPreference: Int): Int {
            return boxStart + (boxEnd - boxStart) / 2 - (viewStart + (viewEnd - viewStart) / 2)
        }

    }



    override fun renderViewEffect(viewEffect: AnnotationViewEffect) {

        UniDirectionMVI.renderViewEffect(this, viewEffect)
        AutoAnalysisMVI.renderViewEffect(this, viewEffect)
        CopyAutoAnnotationMVI.renderViewEffect(this, viewEffect)
        PlaybackMVI.renderViewEffect(this, viewEffect)
        FetchNewsFolderMVI.renderViewEffect(this, viewEffect)
        FetchNewsFileMVI.renderViewEffect(this, viewEffect)
        SetToolUsingMVI.renderViewEffect(this, viewEffect)
        SaveDataMVI.renderViewEffect(this, viewEffect)

        // almost using show toast message
        when (viewEffect) {
            is AnnotationViewEffect.ShowToast -> toast(message = viewEffect.message)
        }
    }

    fun showUserLoginDialog() {
        val dlg = LoginDialog(this, this)
        dlg.setCanceledOnTouchOutside(false)
        dlg.show()
    }

    fun showSaveDialog(file: String, bitmaps: List<Bitmap>, dicomAnnotation: DicomAnnotation, dicomDiagnosis: DicomDiagnosis) {
        val dlg = SaveDialog(
            this,
            this,
            file,
            bitmaps,
            dicomAnnotation,
            dicomDiagnosis
        )
        dlg.setCanceledOnTouchOutside(false)
        dlg.show()
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

    override fun onDiagnosisEntered(dicomDiagnosis: DicomDiagnosis) {
        SaveDataMVI.process(viewModel, AnnotationViewEvent.SaveDiagnosis(dicomDiagnosis))

        SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveDataToDisk())


    }

    override fun onSaveConfirmed(file: String, dicomAnnotation: DicomAnnotation, dicomDiagnosis: DicomDiagnosis) {
        Log.w(TAG, "Save confirmed")
        SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveConfirmed)
    }



    override fun onSaveLogin(user: User) {
        Log.w(TAG, "onSaveLogin confirmed")
        SaveDataMVI.process(viewModel, AnnotationViewEvent.OnSaveUserLogin(user))
        // BAD DESIGN
        User.remember(this, user.phone)
    }

    override fun onSaveCopyAllFrames() {
        CopyAutoAnnotationMVI.process(viewModel, AnnotationViewEvent.CopyAutoAllFrames)
    }


}