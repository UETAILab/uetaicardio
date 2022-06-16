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

import aacmvi.AacMviActivity
import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.drawable.Drawable
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.view.MotionEvent
import android.widget.SeekBar
import android.widget.SeekBar.OnSeekBarChangeListener
import androidx.activity.viewModels
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.interpretation.InterpretationActVM.Companion.TOOL_DRAW_BOUNDARY
import com.ailab.aicardiotrainer.interpretation.InterpretationActVM.Companion.TOOL_DRAW_POINT
import com.ailab.aicardiotrainer.interpretation.InterpretationActVM.Companion.TOOL_MEASURE_AREA
import com.ailab.aicardiotrainer.interpretation.InterpretationActVM.Companion.TOOL_MEASURE_LENGTH
import com.ailab.aicardiotrainer.studyscreen.DicomPreviewDialog
import com.ailab.aicardiotrainer.toast
import kotlinx.android.synthetic.main.activity_interpretation.*
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader

class InterpretationActivity : AacMviActivity<InterpretationViewState, InterpretationViewEffect, InterpretationViewEvent, InterpretationActVM>(),
    OnDrawListener,
    OnNormalizeTouchListener,
    OnSopInstanceUIDItemClicked, OnToolUsing {

    companion object {
        const val TAG = "InterpretationActivity"
        const val MY_PERMISSIONS_REQUEST_CODE = 1
        const val INTENT_STUDY_INSTANCE_UID = "STUDY_INSTANCE_UID"
        const val INTENT_FILE_MP4_PATH = "FILE_MP4_PATH"

        var bitmapHeart : Bitmap? = null
        var bitmapPlay : Bitmap? = null
        var bitmapPause : Bitmap? = null
        var bitmapBullEye : Bitmap? = null
        fun createIntent(context: Context, studyInstanceUID: String, fileMP4Path: String): Intent {
            val intent = Intent(context, InterpretationActivity::class.java)
            intent.putExtra(INTENT_STUDY_INSTANCE_UID, studyInstanceUID)
            intent.putExtra(INTENT_FILE_MP4_PATH, fileMP4Path)
            return intent
        }

    }
    val handle =  Handler()

    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
//                    mOpenCvCameraView.enableView()
//                    mOpenCvCameraView.setOnTouchListener(this@MainActivity)
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    val interpretationFrameRVAdapter by lazy {
        InterpretationFrameRVAdapter(listener = {

            viewModel.process(InterpretationViewEvent.SopInstanceUIDFrameClicked(it.tag as FrameItem))


        }, longListener = {

            viewModel.process(InterpretationViewEvent.SopInstanceUIDFrameLongClicked(it.tag as FrameItem))

            true
        }, isVertical = true, interpretationActVM = viewModel)
    }



    override fun onResume() {
        super.onResume()

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }


    }

    private fun pushVideoToCanvas(handler: Handler) {
        handler.postDelayed({
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.NextFrame)
            pushVideoToCanvas(handler)
        }, 30L)
    }

    val studyRepresentationGVAdapter by lazy {
        StudyRepresentationGVAdapter(this, this)
    }


    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        inter_iv_draw_canvas.saveToBundle(outState)
    }

    override fun onRestoreInstanceState(savedInstanceState: Bundle) {
        super.onRestoreInstanceState(savedInstanceState)
        inter_iv_draw_canvas.getFromBundle(savedInstanceState)
    }

    override fun onStart() {
        super.onStart()

        viewModel.viewStates().value?.let {
            Log.w(TAG, "onStart Status: ${it.status.javaClass.name}")
        }
        pushVideoToCanvas(handle)

        // render gridview of list file dicom
        viewModel.viewStates().value?.let {
            Log.w(TAG, "GET studyRepresentationGVAdapter:  ${it.studyFiles}")
            studyRepresentationGVAdapter.submitList(it.getListSopInstanceUIDItem())
        }

        // render frame rv adapter of SopinstanceUID (dicom file)
        viewModel.viewStates().value?.let {
            Log.w(TAG, "GET interpretationFrameRVAdapter: ${it.getSubmitListFrameItem().size}")

            interpretationFrameRVAdapter.submitList(it.getSubmitListFrameItem())
        }
        // render canvas
        viewModel.getRenderMP4FrameObject()?.let {
            InterpretationPlaybackMVI.renderViewEffect(this, InterpretationViewEffect.RenderMP4Frame(it))
        }

        // render component view in activity
        InterpretationPlaybackMVI.renderViewEffect(this, InterpretationViewEffect.RenderComponentActivity(idComponent = R.id.bt_play_pause, isPlaying = viewModel.getIsPlaying()))


    }

    override fun onStop() {
        super.onStop()
        handle.removeCallbacksAndMessages(null)
    }


    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_interpretation)

        checkAndRequestPermissions(this, arrayOf(
            Manifest.permission.INTERNET,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        ))



        inter_gv_dicom_preview.apply {
            adapter = studyRepresentationGVAdapter
        }
        inter_iv_draw_canvas.setOnDrawListener(this)
        inter_iv_draw_canvas.setOnNormalizeTouchListener(this)


        inter_rv_frames.adapter = interpretationFrameRVAdapter


        bt_tool_interpretation.setOnClickListener {
            // Test dialog tool
            showEditingToolDialog()
        }


        bt_prev_frame.setOnLongClickListener {
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowFirstFrame)
            true
        }

        bt_next_frame.setOnLongClickListener{
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowLastFrame)
            true
        }

        bt_next_frame.setOnClickListener {
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowNextFrame)
        }

        bt_prev_frame.setOnClickListener {
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowPreviousFrame)
        }

        bt_play_pause.setOnClickListener {
            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.PlayPauseVideo)
        }

        bt_undo.setOnClickListener {
            ProcessUndoAndClearMVI.process(viewModel, InterpretationViewEvent.UndoAnnotation)
        }

        bt_clear.setOnClickListener {
            ProcessUndoAndClearMVI.process(viewModel, InterpretationViewEvent.ClearAnnotation)
        }


        inter_seek_bar.setOnSeekBarChangeListener(
            object: OnSeekBarChangeListener {
                override fun onProgressChanged(seek: SeekBar, progress: Int, fromUser: Boolean) {
//                    InterpretationEnhanceBitmapMVI.process(viewModel, InterpretationViewEvent.EnhanceContrastBitmap(threshold=seek.progress))

                    viewModel.process(InterpretationViewEvent.OnChangeGammaCorrection(threshold=seek.progress))

                    renderTextViewGammaCorrection()


                }
                override fun onStartTrackingTouch(seek: SeekBar) {}

                override fun onStopTrackingTouch(seek: SeekBar) {
//                    Log.w(TAG, "Progress: ${seek.progress}")
//                    InterpretationEnhanceBitmapMVI.process(viewModel, InterpretationViewEvent.EnhanceContrastBitmap(threshold=seek.progress))
                }
            }
        )
        // NOTE: su dung chung toolTypeClick (Shot, long clicked) for checkbox value
        cb_perimeter.setOnClickListener {
            SetToolUsingMVI.process(viewModel, InterpretationViewEvent.OnToolUsing(toolName = InterpretationActVM.TOOL_CHECK_BOX_PERIMETER,
                toolButtonID = R.id.cb_perimeter,
                toolTypeClick = cb_perimeter.isChecked,
                isPlaying = viewModel.getIsPlaying()))
        }

        cb_esv.setOnClickListener {
            SetToolUsingMVI.process(viewModel, InterpretationViewEvent.OnToolUsing(toolName = InterpretationActVM.TOOL_CHECK_BOX_ESV,
                toolButtonID = R.id.cb_esv,
                toolTypeClick = cb_esv.isChecked,
                isPlaying = viewModel.getIsPlaying()))
        }
        cb_edv.setOnClickListener {
            SetToolUsingMVI.process(viewModel, InterpretationViewEvent.OnToolUsing(toolName = InterpretationActVM.TOOL_CHECK_BOX_EDV,
                toolButtonID = R.id.cb_edv,
                toolTypeClick = cb_edv.isChecked,
                isPlaying = viewModel.getIsPlaying()))
        }

        bt_tool_bull_eye.setOnClickListener {
            Log.w(TAG, "Clicked bt_tool_bull_eye")
            BullEyeMappingMVI.process(viewModel, InterpretationViewEvent.ShowBullEyeMapping)
        }


    }

    fun renderTextViewGammaCorrection() {
        InterpretationPlaybackMVI.renderViewEffect(this, InterpretationViewEffect.RenderTextViewGammaCorrection(viewModel.getGammaValue()))
    }


    override val viewModel: InterpretationActVM by viewModels()

    override fun renderViewState(viewState: InterpretationViewState) {

        StudyRepresentationMVI.renderViewState(this, viewState)
        BullEyeMappingMVI.renderViewState(this, viewState)
//        SopInstanceUIDMVI.renderViewState(this, viewState)
//        InterpretationEnhanceBitmapMVI.renderViewState(this, viewState)

        if (bitmapHeart == null) {
            bitmapHeart = BitmapFactory.decodeResource(this.resources, R.drawable.heart)
        }
        Log.w(TAG, "renderViewState: ${viewState.studyFiles}")
        when(viewState.status) {
            InterpretationViewStatus.Start -> {

//                val studyInstanceUID = "1.2.840.113619.2.300.7348.1516976410.0.4"
                val fileMP4Path = "/storage/emulated/0/Download/000005/1.2.840.113619.2.300.7348.1516976410.0.46.512____I1QF5EG2.mp4"
                val studyInstanceUID = "000005"
                StudyRepresentationMVI.process(viewModel, InterpretationViewEvent = InterpretationViewEvent.PlayBackMP4File(fileMP4Path=fileMP4Path))

//                LoadingRepresentationStudyInstanceUID
                StudyRepresentationMVI.process(viewModel, InterpretationViewEvent.LoadingRepresentationStudyInstanceUID(studyInstanceUID=studyInstanceUID))

            }
        }
    }

    override fun renderViewEffect(viewEffect: InterpretationViewEffect) {
//        Log.w(TAG, "renderViewEffect: ${viewEffect}")
        InterpretationPlaybackMVI.renderViewEffect(this, viewEffect)
        SetToolUsingMVI.renderViewEffect(this, viewEffect)

        when (viewEffect) {
            is InterpretationViewEffect.ShowToast -> toast(message = viewEffect.message)
        }

    }

    override fun onTouchEvent(view: InterpretationCanvasView, event: MotionEvent?, ix: Float, iy: Float) {
        Log.w(TAG, "onTouchEvent ${event?.action} ${ix} ${iy}")
        ProcessTouchEventMVI.process(viewModel, InterpretationViewEvent.ProcessTouchEvent(view, event, ix, iy))

    }

    override fun draw(view: InterpretationCanvasView, canvas: Canvas?) {
//        TODO("Not yet implemented")
        RenderDrawMVI.process(viewModel, InterpretationViewEvent.RenderDraw(view, canvas, enableManualDraw = viewModel.getEnableManualDraw(), enableAutoDraw = viewModel.getEnableAutoDraw()))

    }


    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        when (requestCode) {
            MY_PERMISSIONS_REQUEST_CODE -> {
                permissions.forEach { permission ->
                    Log.w(TAG, "onRequestPermissionsResult $permission")
                }
            }
        }
    }

    override fun onSopInstanceUIDItemClicked(item: SopInstanceUIDItem) {
        Log.w(TAG, "onSopInstanceUIDItemClicked ${item}")

    }

    override fun onSopInstanceUIDItemLongClicked(item: SopInstanceUIDItem) {
        Log.w(TAG, "onSopInstanceUIDItemLongClicked ${item}")
        SopInstanceUIDMVI.process(viewModel, InterpretationViewEvent.LoadingMP4SopInstanceUID(item))

    }

    var editingToolDialog: EditingToolDialog? = null
    fun showEditingToolDialog() {
        editingToolDialog = EditingToolDialog(this, this)
        editingToolDialog!!.setCanceledOnTouchOutside(true)
        editingToolDialog!!.show()
    }

    fun closeEditingToolDialog() {
        editingToolDialog?.let {
            it.dismiss()
            editingToolDialog = null
        }
    }

    @SuppressLint("ResourceAsColor")
    override fun onToolSelected(toolName: String, toolTypeClick: Boolean, toolButtonID: Int) {
        SetToolUsingMVI.process(viewModel, InterpretationViewEvent.OnToolUsing(toolName, toolTypeClick, toolButtonID))
    }


    fun showBullEyeMappingDialog(gls_array: List<Float>) {
        if (bitmapBullEye == null) {
            bitmapBullEye = BitmapFactory.decodeResource(this.resources, R.drawable.bull_eye_mapping)
        }

        bitmapBullEye?.let {
            val dlg =  BullEyeMappingDialog(this, gls_array , bitmap= it)
            dlg.setCanceledOnTouchOutside(false)
            dlg.show()
        }
    }

}