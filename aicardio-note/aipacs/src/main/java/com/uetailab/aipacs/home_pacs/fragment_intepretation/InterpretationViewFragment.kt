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

import aacmvi.AacMviFragment
import android.app.Activity
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.fragment.app.viewModels
import com.uetailab.aipacs.R
import com.uetailab.aipacs.home_pacs.HomePacsAPI
import com.uetailab.aipacs.home_pacs.HomePacsActivity
import com.uetailab.aipacs.home_pacs.toast
import kotlinx.android.synthetic.main.fragment_home_view_dialog_progress.*
import kotlinx.android.synthetic.main.fragment_interpretation_view.*
import kotlinx.android.synthetic.main.interpretation_checkbox_tool.*
import kotlinx.android.synthetic.main.interpretation_editing_tool.*
import kotlinx.android.synthetic.main.interpretation_playback_tool.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch

class InterpretationViewFragment: AacMviFragment<InterpretationViewState, InterpretationViewEffect, InterpretationViewEvent, InterpretationViewVM>(),
    HomePacsAPI.ProgressDownloadListener,
    OnStudyPreviewClicked,
    OnDrawListener, OnNormalizeTouchListener,
    InterpretationViewDiagnosisDialog.OnDiagnosisEnteredListener {
    companion object {
        const val TAG = "InterpretationViewFragment"
        var bitmapPlay: Bitmap? = null
        var bitmapPause: Bitmap? = null
        var bitmapBullEye: Bitmap? = null
    }
    interface OnInterpretationViewVMPass {
        fun onInterpretationViewVMPass(viewModel: InterpretationViewVM)
    }

    lateinit var dataPasser: OnInterpretationViewVMPass
    override fun onAttach(context: Context) {
        super.onAttach(context)
        dataPasser = context as OnInterpretationViewVMPass
    }

    fun passData(data: String){
        dataPasser.onInterpretationViewVMPass(viewModel)
    }

    val listTools: ArrayList<ImageView> = ArrayList()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        super.onCreateView(inflater, container, savedInstanceState)
        val view = inflater.inflate(R.layout.fragment_interpretation_view, container, false)
        return view
    }

    val studyGVAdapter by lazy {
        InterpretationViewStudyGVAdapter(context as Context, this, viewModel)
    }
    val frameCanvasRVAdapter by lazy {
        InterpretationViewFrameCanvasRVAdapter(listener = {
            viewModel.process(InterpretationViewEvent.FrameCanvasClicked(it.tag as FrameCanvasItem)) },
            longListener = { viewModel.process(InterpretationViewEvent.FrameCanvasLongClicked(it.tag as FrameCanvasItem)); true },
            interpretationViewVM = viewModel)
    }

    val downloadDataStudyMVIListener = InterpretationViewDownloadDataMVIListener(this)
    val autoCalculatorMVIListener = InterpretationViewToolCalculatorMVIListener(this)

    val handle =  Handler()


//    override fun onSaveInstanceState(outState: Bundle) {
//        super.onSaveInstanceState(outState)
//        interpretation_draw_canvas.saveToBundle(outState)
//    }
//
//    override fun onRestoreInstanceState(savedInstanceState: Bundle) {
//        super.onRestoreInstanceState(savedInstanceState)
//        interpretation_draw_canvas.getFromBundle(savedInstanceState)
//    }

    private fun pushVideoToCanvas(handler: Handler) {
        handler.postDelayed({
            // just like click into button next_frame, tuy nhien can phan biet la trang thai hien tai co play ko thong qua isPlayingNextFrame
            if (viewModel.getIsPlaying())
                InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.RenderFragmentView(buttonID=R.id.bt_next_frame, isLongClicked=false, isPlayingNextFrame = true))
            pushVideoToCanvas(handler)

        }, 30L)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {

        when((activity as HomePacsActivity).checkIsNewStudy()) {
            true -> {
                (activity as HomePacsActivity).getHomeViewVM()?.let {
                    viewModel.onPassDataFromHomeView(it)
                }
            }
            false -> {
                (activity as HomePacsActivity).getInterpretationViewVM()?.let {
                    viewModel.onPassDataFromInterpretationView(it)
                }
            }
        }




        interpretation_gv_study_preview.apply {
            adapter = studyGVAdapter
        }

        interpretation_rv_frame_draw_canvas_item.adapter = frameCanvasRVAdapter


        interpretation_draw_canvas.setOnDrawListener(this)
        interpretation_draw_canvas.setOnNormalizeTouchListener(this)


        // Init editing tool button
        listTools.add(bt_tool_diagnosis)
        listTools.add(bt_tool_zoom)
        listTools.add(bt_tool_draw_point)
        listTools.add(bt_tool_draw_boundary)
        listTools.add(bt_tool_measure_length)
        listTools.add(bt_tool_measure_area)
        listTools.add(bt_tool_ef_calculate)
        listTools.add(bt_tool_bull_eye)
        listTools.add(bt_no_action)


        bt_tool_diagnosis.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickDiagnosis(imageView = bt_tool_diagnosis)))
        }
        bt_tool_diagnosis.setOnLongClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickDiagnosis(imageView = bt_tool_diagnosis, isLongClicked = true)))
            true
        }

        bt_tool_zoom.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickZooming(imageView = bt_tool_zoom)))
        }
        bt_tool_zoom.setOnLongClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickZooming(imageView = bt_tool_zoom, isLongClicked = true)))
            true
        }


        bt_tool_draw_point.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickDrawPoint(imageView = bt_tool_draw_point)))
        }
        bt_tool_draw_point.setOnLongClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickDrawPoint(imageView = bt_tool_draw_point, isLongClicked = true)))
            true
        }


        bt_tool_draw_boundary.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickDrawBoundary(imageView = bt_tool_draw_boundary)))
        }
        bt_tool_draw_boundary.setOnLongClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickDrawBoundary(imageView = bt_tool_draw_boundary, isLongClicked = true)))
            true
        }


        bt_tool_measure_length.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickMeasureLength(imageView = bt_tool_measure_length)))
        }
        bt_tool_measure_length.setOnLongClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickMeasureLength(imageView = bt_tool_measure_length, isLongClicked = true)))
            true
        }


        bt_tool_measure_area.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickMeasureArea(imageView = bt_tool_measure_area)))
        }
        bt_tool_measure_area.setOnLongClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickMeasureArea(imageView = bt_tool_measure_area, isLongClicked = true)))
            true
        }

        bt_tool_ef_calculate.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickMeasureEF(imageView = bt_tool_ef_calculate)))

            InterpretationViewCalculatorToolMVI.process(viewModel, InterpretationViewEvent.OnMeasureEFManual)

//            showEFCalculationDialog()

        }
        bt_tool_ef_calculate.setOnLongClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickMeasureEF(imageView = bt_tool_ef_calculate, isLongClicked = true)))
            true
        }



        bt_tool_bull_eye.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickShowBullEye(imageView = bt_tool_bull_eye)))
//            autoCalculatorMVIListener.process(viewModel, InterpretationViewEvent.OnAutoCalculateEFGLS)
            // show 18 bars chart
            showGLSBarChart()

        }
        bt_tool_bull_eye.setOnLongClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickShowBullEye(imageView = bt_tool_bull_eye, isLongClicked = true)))
            showGLSBullEyeMapping()
            true
        }

        bt_tool_auto_server.setOnClickListener {
            autoCalculatorMVIListener.process(viewModel, InterpretationViewEvent.OnAutoCalculateEFGLS)
        }

        bt_no_action.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickNoAction(imageView = bt_no_action)))
        }
        bt_no_action.setOnLongClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnToolClicked(InterpretationViewTool.OnClickNoAction(imageView = bt_no_action, isLongClicked = true)))
            true
        }

        // Done init editing tool button

        bt_prev_frame.setOnLongClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.RenderFragmentView(buttonID=R.id.bt_prev_frame, isLongClicked=true))
            true
        }

        bt_next_frame.setOnLongClickListener{
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.RenderFragmentView(buttonID=R.id.bt_next_frame, isLongClicked=true, isPlayingNextFrame = false))
            true
        }

        bt_next_frame.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.RenderFragmentView(buttonID=R.id.bt_next_frame, isLongClicked=false, isPlayingNextFrame = false))
        }

        bt_prev_frame.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.RenderFragmentView(buttonID=R.id.bt_prev_frame, isLongClicked=false))
        }

        bt_play_pause.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.RenderFragmentView(buttonID=R.id.bt_play_pause, isLongClicked=false))
        }

//        bt_no_action.setOnClickListener {
//            interpretation_gv_study_preview.isGone = true
//        }

        cb_perimeter.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnCheckBoxClicked(InterpretationViewTool.OnClickCheckBox(checkBox = cb_perimeter)))
        }

        cb_esv.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnCheckBoxClicked(InterpretationViewTool.OnClickCheckBox(checkBox = cb_esv)))
        }

        cb_edv.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnCheckBoxClicked(InterpretationViewTool.OnClickCheckBox(checkBox = cb_edv)))
        }

        bt_undo.setOnClickListener {
            InterpretationViewTouchEventMVI.process(viewModel, InterpretationViewEvent.OnClickUndoClearDataAnnotation(isClear = false) )
        }

        bt_clear.setOnClickListener {
            InterpretationViewTouchEventMVI.process(viewModel, InterpretationViewEvent.OnClickUndoClearDataAnnotation(isClear = true) )

        }

        bt_save_data.setOnClickListener {
            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnSaveDataAnnotationToServer)
        }

        cb_auto_ef.setOnClickListener {
            viewModel.setAutoDraw(cb_auto_ef.isChecked)
//            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.RenderFragmentView(buttonID=R.id.bt_play_pause, isLongClicked=false))

        }
        cb_draw_manual.setOnClickListener {
            viewModel.setManualDraw(cb_draw_manual.isChecked)
//            InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.RenderFragmentView(buttonID=R.id.bt_play_pause, isLongClicked=false))
        }

    }

    override fun onStart() {
        super.onStart()
        pushVideoToCanvas(handle)
    }
    override fun onStop() {
        super.onStop()
        handle.removeCallbacksAndMessages(null)
    }



    override val viewModel: InterpretationViewVM by viewModels()
//        get() = TODO("Not yet implemented")


    override fun renderViewState(viewState: InterpretationViewState) {

        InterpretationViewCalculatorToolMVI.renderViewState(this, viewState)
        when (viewState.status) {
            InterpretationViewStatus.PassedDataFromHomeView, InterpretationViewStatus.PassedDataFromInterpretationView -> {
                Log.w(TAG, "renderViewState -- InterpretationViewStatus.PassedDataFromHomeView")
                studyGVAdapter.submitList(viewState.getListStudyGVItem())
                frameCanvasRVAdapter.submitList(viewState.getListFrameCanvasRVItem())
                frameCanvasRVAdapter.setCurrentPosition(viewModel.getCurrentFrameIndex())


                val currentBitmap = viewModel.getCurrentFrameBitmap()
                interpretation_draw_canvas.setFitScale(currentBitmap)
                // Luu y: xem xem tai sao khi passdata lai phai setCustomBitmap, neu ko xet thi man hinh trang
                // trong khi do thi InterpretationViewEffect.RenderFragmentView da thuc hien nhiem vu get frame hien thi
                interpretation_draw_canvas.setCustomImageBitmap(currentBitmap)
            }

            is InterpretationViewStatus.OnFetchingData -> {
                openProgressDialog()
            }
            is InterpretationViewStatus.FetchedErrorData -> {
                closeProgressDialog()
            }

            is InterpretationViewStatus.FetchedData -> {
                closeProgressDialog()

                dataPasser.onInterpretationViewVMPass(viewModel)

                when (viewState.status.viewEvent) {
                    is InterpretationViewEvent.FetchFileMP4 -> {
                        if (viewState.bitmaps.size > 0) {
                            frameCanvasRVAdapter.submitList(viewState.getListFrameCanvasRVItem())
                            frameCanvasRVAdapter.setCurrentPosition(viewModel.getCurrentFrameIndex())
                            // lan dau load xong video mp4 thi set trang thai playing = True
//                            viewModel.setIsPlaying(true)
//                            Log.w(TAG, "InterpretationViewEvent.FetchFileMP4 ${viewModel.getTextDrawCanvas()} ${viewState.bitmaps.size}")
                            interpretation_draw_canvas.setFitScale(viewState.bitmaps.get(0))
                            interpretation_draw_canvas.setCustomImageBitmap(viewState.bitmaps.get(0))
                            interpretation_draw_canvas.infoText = viewModel.getTextDrawCanvas()
                        }
                    }

                }

            }
        }
    }

    override fun renderViewEffect(viewEffect: InterpretationViewEffect) {
        InterpretationViewComponentClickedMVI.renderViewEffect(this, viewEffect = viewEffect)
        when (viewEffect) {
            is InterpretationViewEffect.ShowToast -> {
                activity?.let {
                    it.toast(viewEffect.message)
                }
            }
        }
    }


    override fun onStudyPreviewClicked(item: StudyGVItem) {
//        TODO("Not yet implemented")
        Log.w(TAG, "${item} ${item.name}")
        val relativePath = viewModel.getRelativePath(item.name)
        if (viewModel.studyID != null && viewModel.studyInstanceUID != null && relativePath != null)
            downloadDataStudyMVIListener.process(viewModel, InterpretationViewEvent.FetchFileMP4(viewModel.studyID!!, viewModel.studyInstanceUID!!, relativePath ))
    }

    override fun onStudyPreviewLongClicked(item: StudyGVItem): Boolean {
//        TODO("Not yet implemented")
        return true
    }


    fun showEFCalculationDialog() {
        val dlg = InterpretationViewEFCalculationDialog(activity as Activity, viewModel)
        dlg.setCanceledOnTouchOutside(false)
        dlg.show()
    }

    fun showGLSBarChart() {
        val dlg = InterpretationViewGLSBarChartDialog(activity as Activity, viewModel)
        dlg.setCanceledOnTouchOutside(false)
        dlg.show()
    }

    fun showGLSBullEyeMapping() {
        if (bitmapBullEye == null) {
            bitmapBullEye = BitmapFactory.decodeResource(this.resources, R.drawable.bull_eye_mapping)
        }

        bitmapBullEye?.let {
            val dlg = InterpretationViewGLSBullEyeMappingDialog(activity as Activity, viewModel, it)
            dlg.setCanceledOnTouchOutside(false)
            dlg.show()
        }


    }

    fun showDiagnosisDialog(dicomDiagnosis: DicomDiagnosis) {
        val dlg = InterpretationViewDiagnosisDialog(
            activity as Activity,
            this,
            dicomDiagnosis,
            viewModel
        )
        dlg.setCanceledOnTouchOutside(false)
        dlg.show()
    }


    var progressDialog : InterpretationViewProgressDialog? = null

    fun openProgressDialog() {
        progressDialog = InterpretationViewProgressDialog(activity as Activity)
        progressDialog!!.setCanceledOnTouchOutside(false)
        progressDialog!!.show()
    }

    fun closeProgressDialog() {
        progressDialog?.let {
            it.dismiss()
            progressDialog = null
        }
    }

    override fun update(bytesRead: Long, contentLength: Long, done: Boolean) {
        Log.w(TAG, "Process to: ${bytesRead} ${contentLength} DONE: ${(100 * bytesRead) / contentLength} %")
        GlobalScope.launch(Dispatchers.Main) {
//            if (done) closeProgressDialog()
            progressDialog?.let {
                it.tv_progress_percentage.text = ((100 * bytesRead) / contentLength).toString()
                it.pb_progress.progress = ((100 * bytesRead) / contentLength).toInt()
            }
        }
    }

    override fun draw(view: InterpretationViewStudyPreviewCanvasView, canvas: Canvas?) {
        InterpretationViewDrawTouchEventMVI.process(viewModel, InterpretationViewEvent.RenderTouchDraw(view, canvas, enableManualDraw = viewModel.getEnableManualDraw(), enableAutoDraw = viewModel.getEnableAutoDraw()))
    }

    override fun onTouchEvent(view: InterpretationViewStudyPreviewCanvasView, event: MotionEvent?, ix: Float, iy: Float) {
        Log.w(TAG, "onTouchEvent: ${ix} ${iy}")
        InterpretationViewTouchEventMVI.process(viewModel, InterpretationViewEvent.OnTouchEvent(view, event, ix, iy))
    }

    override fun onDiagnosisEntered(dicomDiagnosis: DicomDiagnosis) {
        Log.w(TAG, "onDiagnosisEntered ${dicomDiagnosis}")
        InterpretationViewComponentClickedMVI.process(viewModel, InterpretationViewEvent.OnSaveDataDiagnosis(dicomDiagnosis))

    }

}