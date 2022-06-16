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

package com.ailab.aicardiotrainer.nav_interpretation.ui.annotation
//
//import aacmvi.AacMviActivity
//import android.Manifest
//import android.graphics.Bitmap
//import android.graphics.BitmapFactory
//import android.graphics.Canvas
//import android.os.Bundle
//import android.os.Handler
//import android.util.Log
//import android.view.MotionEvent
//import android.widget.SeekBar
//import android.widget.SeekBar.OnSeekBarChangeListener
//import androidx.activity.viewModels
//import com.ailab.aicardiotrainer.R
//import com.ailab.aicardiotrainer.toast
//import kotlinx.android.synthetic.main.activity_interpretation.*
//import org.opencv.android.BaseLoaderCallback
//import org.opencv.android.LoaderCallbackInterface
//import org.opencv.android.OpenCVLoader
//
//class InterpretationActivity : AacMviActivity<InterpretationViewState, InterpretationViewEffect, InterpretationViewEvent, InterpretationActVM>(),
//    OnDrawListener,
//    OnNormalizeTouchListener,
//    OnSopInstanceUIDItemClicked {
//
//    companion object {
//        const val TAG = "InterpretationActivity"
//        const val MY_PERMISSIONS_REQUEST_CODE = 1
//
//        var bitmapHeart : Bitmap? = null
//        var bitmapPlay : Bitmap? = null
//        var bitmapPause : Bitmap? = null
//
//    }
//    val handle =  Handler()
//
//    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
//        override fun onManagerConnected(status: Int) {
//            when (status) {
//                SUCCESS -> {
//                    Log.i(TAG, "OpenCV loaded successfully")
////                    mOpenCvCameraView.enableView()
////                    mOpenCvCameraView.setOnTouchListener(this@MainActivity)
//                }
//                else -> {
//                    super.onManagerConnected(status)
//                }
//            }
//        }
//    }
//
//    val interpretationFrameRVAdapter by lazy {
//        InterpretationFrameRVAdapter(listener = {
//
//            viewModel.process(InterpretationViewEvent.SopInstanceUIDFrameClicked(it.tag as FrameItem))
//
//
//        }, longListener = {
//
//            viewModel.process(InterpretationViewEvent.SopInstanceUIDFrameLongClicked(it.tag as FrameItem))
//
//            true
//        }, isVertical = true, interpretationActVM = viewModel)
//    }
//
//
//    override fun onResume() {
//        super.onResume()
//        if (!OpenCVLoader.initDebug()) {
//            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
//            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
//        } else {
//            Log.d(TAG, "OpenCV library found inside package. Using it!");
//            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
//        }
//
//
//    }
//
//    private fun pushVideoToCanvas(handler: Handler) {
//        handler.postDelayed({
//            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.NextFrame)
//            pushVideoToCanvas(handler)
//        }, 30L)
//    }
//
//    val studyRepresentationGVAdapter by lazy {
//        StudyRepresentationGVAdapter(this, this)
//    }
//
//
//    override fun onSaveInstanceState(outState: Bundle) {
//        super.onSaveInstanceState(outState)
//        inter_iv_draw_canvas.saveToBundle(outState)
//    }
//
//    override fun onRestoreInstanceState(savedInstanceState: Bundle) {
//        super.onRestoreInstanceState(savedInstanceState)
//        inter_iv_draw_canvas.getFromBundle(savedInstanceState)
//    }
//
//    override fun onStart() {
//        super.onStart()
//
//        pushVideoToCanvas(handle)
//
//        // render gridview of list file dicom
//        viewModel.viewStates().value?.let {
//            studyRepresentationGVAdapter.submitList(it.getListSopInstanceUIDItem())
//        }
//
//        // render frame rv adapter of SopinstanceUID (dicom file)
//        viewModel.viewStates().value?.let {
//            interpretationFrameRVAdapter.submitList(it.getSubmitListFrameItem())
//        }
//        // render canvas
//        viewModel.getRenderMP4FrameObject()?.let {
//            InterpretationPlaybackMVI.renderViewEffect(this, InterpretationViewEffect.RenderMP4Frame(it))
//        }
//
//        // render component view in activity
//        InterpretationPlaybackMVI.renderViewEffect(this, InterpretationViewEffect.RenderComponentActivity(idComponent = R.id.bt_play_pause, isPlaying = viewModel.getIsPlaying()))
//
//
//    }
//
//    override fun onStop() {
//        super.onStop()
//        handle.removeCallbacksAndMessages(null)
//    }
//
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//
//        super.onCreate(savedInstanceState)
//        setContentView(R.layout.activity_interpretation)
//
//        checkAndRequestPermissions(this, arrayOf(
//            Manifest.permission.INTERNET,
//            Manifest.permission.READ_EXTERNAL_STORAGE,
//            Manifest.permission.WRITE_EXTERNAL_STORAGE
//        ))
//
//
//
//        inter_gv_dicom_preview.apply {
//            adapter = studyRepresentationGVAdapter
//        }
//        inter_iv_draw_canvas.setOnDrawListener(this)
//        inter_iv_draw_canvas.setOnNormalizeTouchListener(this)
//
//
//        inter_rv_frames.adapter = interpretationFrameRVAdapter
//
//
//        bt_prev_frame.setOnLongClickListener {
//            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowFirstFrame)
//            true
//        }
//
//        bt_next_frame.setOnLongClickListener{
//            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowLastFrame)
//            true
//        }
//
//        bt_next_frame.setOnClickListener {
//            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowNextFrame)
//        }
//
//        bt_prev_frame.setOnClickListener {
//            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.ShowPreviousFrame)
//        }
//
//        bt_play_pause.setOnClickListener {
//            InterpretationPlaybackMVI.process(viewModel, InterpretationViewEvent.PlayPauseVideo)
//        }
//
//
//        inter_seek_bar.setOnSeekBarChangeListener(
//            object: OnSeekBarChangeListener {
//                override fun onProgressChanged(seek: SeekBar, progress: Int, fromUser: Boolean) {
////                    InterpretationEnhanceBitmapMVI.process(viewModel, InterpretationViewEvent.EnhanceContrastBitmap(threshold=seek.progress))
//
//                    viewModel.process(InterpretationViewEvent.OnChangeGammaCorrection(threshold=seek.progress))
//
//                    renderTextViewGammaCorrection()
//
//
//                }
//                override fun onStartTrackingTouch(seek: SeekBar) {}
//
//                override fun onStopTrackingTouch(seek: SeekBar) {
////                    Log.w(TAG, "Progress: ${seek.progress}")
////                    InterpretationEnhanceBitmapMVI.process(viewModel, InterpretationViewEvent.EnhanceContrastBitmap(threshold=seek.progress))
//                }
//            }
//        )
//
//
//    }
//
//    fun renderTextViewGammaCorrection() {
//        InterpretationPlaybackMVI.renderViewEffect(this, InterpretationViewEffect.RenderTextViewGammaCorrection(viewModel.getGammaValue()))
//    }
//
//
//
//
//
//
//
//
//    override val viewModel: InterpretationActVM by viewModels()
//
//    override fun renderViewState(viewState: InterpretationViewState) {
//
//        StudyRepresentationMVI.renderViewState(this, viewState)
//        SopInstanceUIDMVI.renderViewState(this, viewState)
////        InterpretationEnhanceBitmapMVI.renderViewState(this, viewState)
//
//        if (bitmapHeart == null) {
//            bitmapHeart = BitmapFactory.decodeResource(this.resources, R.drawable.heart)
//        }
//        when(viewState.status) {
//            InterpretationViewStatus.Start -> {
//
////                Log.w(TAG, "Go InterpretationViewStatus.Start ${bitmapHeart?.width} ${bitmapHeart?.height}")
////                inter_iv_draw_canvas.setCustomImageBitmap(bitmapHeart)
//                val studyInstanceUID = "1.2.40.0.13.0.11.2672.5.2013102492.1340595.20130717095716"
//                StudyRepresentationMVI.process(viewModel, InterpretationViewEvent = InterpretationViewEvent.LoadingRepresentationStudyInstanceUID(studyInstanceUID = studyInstanceUID))
//
//            }
//        }
//    }
//
//    override fun renderViewEffect(viewEffect: InterpretationViewEffect) {
////        Log.w(TAG, "renderViewEffect: ${viewEffect}")
//        InterpretationPlaybackMVI.renderViewEffect(this, viewEffect)
//
//        when (viewEffect) {
//            is InterpretationViewEffect.ShowToast -> toast(message = viewEffect.message)
//        }
//
//    }
//
//    override fun onTouchEvent(view: InterpretationCanvasView, event: MotionEvent?, ix: Float, iy: Float) {
//        Log.w(TAG, "onTouchEvent ${event?.action} ${ix} ${iy}")
//    }
//
//    override fun draw(view: InterpretationCanvasView, canvas: Canvas?) {
////        TODO("Not yet implemented")
//    }
//
//
//    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
//        when (requestCode) {
//            MY_PERMISSIONS_REQUEST_CODE -> {
//                permissions.forEach { permission ->
//                    Log.w(TAG, "onRequestPermissionsResult $permission")
//                }
//            }
//        }
//    }
//
//    override fun onSopInstanceUIDItemClicked(item: SopInstanceUIDItem) {
//        Log.w(TAG, "onSopInstanceUIDItemClicked ${item}")
//
//    }
//
//    override fun onSopInstanceUIDItemLongClicked(item: SopInstanceUIDItem) {
//        Log.w(TAG, "onSopInstanceUIDItemLongClicked ${item}")
//        SopInstanceUIDMVI.process(viewModel, InterpretationViewEvent.LoadingMP4SopInstanceUID(item))
//
//
//    }
//
//
//}