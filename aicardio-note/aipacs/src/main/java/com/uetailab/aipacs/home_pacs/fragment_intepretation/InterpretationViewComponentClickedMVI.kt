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

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log
import androidx.lifecycle.viewModelScope
import androidx.recyclerview.widget.LinearSmoothScroller
import androidx.recyclerview.widget.RecyclerView
import com.uetailab.aipacs.R
import com.uetailab.aipacs.home_pacs.LCE
import kotlinx.android.synthetic.main.fragment_interpretation_view.*
import kotlinx.android.synthetic.main.interpretation_checkbox_tool.*
import kotlinx.android.synthetic.main.interpretation_playback_tool.*
import kotlinx.coroutines.launch

class InterpretationViewComponentClickedMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: InterpretationViewComponentClickedMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: InterpretationViewComponentClickedMVI()
                        .also { instance = it }
            }

        const val TAG = "InterpretationViewComponentClickedMVI"

        fun process(interpretationViewVM: InterpretationViewVM, InterpretationViewEvent: InterpretationViewEvent) {
            getInstance().process(interpretationViewVM, InterpretationViewEvent)
        }

        fun renderViewState(interpretationViewFragment: InterpretationViewFragment, viewState: InterpretationViewState) {
            getInstance().renderViewState(interpretationViewFragment, viewState)
        }

        fun renderViewEffect(interpretationViewFragment: InterpretationViewFragment, viewEffect: InterpretationViewEffect) {
            getInstance().renderViewEffect(interpretationViewFragment, viewEffect)
        }
    }

    private val interpretationViewRepository = InterpretationViewRepository.getInstance()

    private fun renderViewEffect(interpretationViewFragment: InterpretationViewFragment, viewEffect: InterpretationViewEffect) {
        when (viewEffect) {
            is InterpretationViewEffect.RenderGridView -> {
                val relativePath = viewEffect.relativePath
                relativePath?.let {
                    interpretationViewFragment.studyGVAdapter.updateItem()
                }
            }
            is InterpretationViewEffect.RenderFragmentView -> {

                if (InterpretationViewFragment.bitmapPlay == null) InterpretationViewFragment.bitmapPlay = BitmapFactory.decodeResource(interpretationViewFragment.resources, R.drawable.ic_play)
                if (InterpretationViewFragment.bitmapPause == null) InterpretationViewFragment.bitmapPause = BitmapFactory.decodeResource(interpretationViewFragment.resources, R.drawable.ic_pause)

                val v = viewEffect.fragmentViewObject
                Log.w(TAG, "InterpretationViewEffect.RenderFragmentView ${v.isPlaying}")


                if (v.numFrame > 0) {
                    interpretationViewFragment.bt_play_pause.setImageBitmap(if (v.isPlaying) InterpretationViewFragment.bitmapPause else InterpretationViewFragment.bitmapPlay)
                    // tai sao khi passing data tu home view fragment ma ko setCustomBitmap thi man  hinh lai hien trang
//                    Log.w(TAG, "renderViewEffect -- InterpretationViewEffect.RenderFragmentView ${v.bitmap?.height} ${v.bitmap?.width}")
                    interpretationViewFragment.interpretation_draw_canvas.infoText = v.frameText
                    interpretationViewFragment.interpretation_draw_canvas.setCustomImageBitmap(v.bitmap)

                    val frameIndex = v.frameID
                    Log.w(TAG, "InterpretationViewEffect.RenderFragmentView ${frameIndex} ${interpretationViewFragment.frameCanvasRVAdapter.getCurrentPosition()}")

                    if (frameIndex != interpretationViewFragment.frameCanvasRVAdapter.getCurrentPosition()) {
//                        Log.w(TAG, "frameIndex != interpretationViewFragment.frameCanvasRVAdapter.getCurrentPosition()")
                        val smoothScroller: RecyclerView.SmoothScroller = CenterSmoothScroller(interpretationViewFragment.interpretation_rv_frame_draw_canvas_item.getContext())
                        smoothScroller.setTargetPosition(frameIndex)
                        interpretationViewFragment.interpretation_rv_frame_draw_canvas_item.layoutManager?.startSmoothScroll(smoothScroller)
                        interpretationViewFragment.frameCanvasRVAdapter.setCurrentPosition(frameIdx=frameIndex)
//                        interpretationViewFragment.frameCanvasRVAdapter.
                    }
                    interpretationViewFragment.frameCanvasRVAdapter.updateFrameViewCanvasView(frameIndex=frameIndex)

                    interpretationViewFragment.cb_edv.isChecked = v.isEDV
                    interpretationViewFragment.cb_esv.isChecked = v.isESV

                }

            }

            is InterpretationViewEffect.RenderToolUsing -> {
                Log.w(TAG, "InterpretationViewEffect.RenderToolUsing")

                when(viewEffect.viewEvent) {

                    is InterpretationViewEvent.OnToolClicked -> {
                        val tool = viewEffect.viewEvent.interpretationViewTool
                        interpretationViewFragment.listTools.forEach {
                            it.background = interpretationViewFragment.activity?.getDrawable(R.drawable.rounded_rectangle)
                        }

                        tool.imageView?.let {
                            if (tool.isLongClicked)  it.background = interpretationViewFragment.activity?.getDrawable(R.drawable.rounded_rectangle_long_clicked)
                            else  it.background = interpretationViewFragment.activity?.getDrawable(R.drawable.rounded_rectangle_short_clicked)
                        }

                        when (tool) {

                            is InterpretationViewTool.OnClickZooming -> {
                                interpretationViewFragment.interpretation_draw_canvas.isZooming = !interpretationViewFragment.interpretation_draw_canvas.isZooming
                            }
                            else -> {
                                interpretationViewFragment.interpretation_draw_canvas.isZooming = false
                            }
                        }

                        // handle diagnosis tool

                        when (tool) {
                            is InterpretationViewTool.OnClickDiagnosis -> {
                                tool.dicomDiagnosis?.let {
                                    interpretationViewFragment.showDiagnosisDialog(it)
                                }

                            }
                        }


                    }

//                    is InterpretationViewEvent.OnCheckBoxClicked -> {
//                        val tool = viewEffect.viewEvent.interpretationViewTool
//
//                    }
                }

            }
        }
    }

    class CenterSmoothScroller(context: Context?) : LinearSmoothScroller(context) {
        override fun calculateDtToFit(viewStart: Int, viewEnd: Int, boxStart: Int, boxEnd: Int, snapPreference: Int): Int {
            return boxStart + (boxEnd - boxStart) / 2 - (viewStart + (viewEnd - viewStart) / 2)
        }
    }

    private fun renderViewState(interpretationViewFragment: InterpretationViewFragment, viewState: InterpretationViewState) {

    }

    fun process(interpretationViewVM: InterpretationViewVM, interpretationViewEvent: InterpretationViewEvent) {
        when (interpretationViewEvent) {
            is InterpretationViewEvent.RenderFragmentView -> {
                interpretationViewVM.viewStates().value?.let {
                    interpretationViewVM.reduce(PlaybackReducer(interpretationViewVM, it, interpretationViewEvent))
                }
            }

            is InterpretationViewEvent.OnToolClicked, is InterpretationViewEvent.OnCheckBoxClicked -> {
                interpretationViewVM.viewStates().value?.let {
                    interpretationViewVM.reduce(EditingToolCheckBoxReducer(interpretationViewVM, it, interpretationViewEvent))
                }
            }

            is InterpretationViewEvent.OnSaveDataDiagnosis -> {
                interpretationViewVM.viewStates().value?.let {
                    interpretationViewVM.reduce(SaveDataDiagnosisReducer(interpretationViewVM, it, interpretationViewEvent))
                }
            }

            is InterpretationViewEvent.OnSaveDataAnnotationToServer -> {
                interpretationViewVM.viewStates().value?.let {
                    interpretationViewVM.reduce(SaveDataAnnotationReducer(interpretationViewVM, it, interpretationViewEvent))
                }
            }
        }

    }


    inner class SaveDataAnnotationReducer(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent)
        : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {
            when(viewEvent) {
                is InterpretationViewEvent.OnSaveDataAnnotationToServer -> {
                    viewModel.viewModelScope.launch {
                        viewModel.viewStates().value?.let {

                            when (val resultLaunch = interpretationViewRepository.saveAnnotationToSever(it)) {
                                is LCE.Result -> {
                                    if (resultLaunch.error) {
                                        // failed to send data to server
                                        viewModel.reduce(InterpretationViewObject(viewState = null, viewEffect = InterpretationViewEffect.ShowToast("Failed when save Data to SERVER")))

                                    } else {
                                        viewModel.reduce(InterpretationViewObject(viewState = null, viewEffect = InterpretationViewEffect.ShowToast("Save Data to SERVER success")))

                                    }
                                }
                            }


                        }
                    }
                }
            }
            return InterpretationViewObject()
        }

    }
    inner class SaveDataDiagnosisReducer(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent)
        : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {
            when(viewEvent) {
                is InterpretationViewEvent.OnSaveDataDiagnosis -> {
                    return InterpretationViewObject(
                        viewState = viewState.copy(status = InterpretationViewStatus.OnSaveDataDiagnosis, dicomInterpretation = updateDicomDiagnosis(viewEvent.dicomDiagnosis)),
                        viewEffect = InterpretationViewEffect.RenderGridView(viewModel.relativePath))
                }
            }
            return InterpretationViewObject()
        }

        fun updateDicomDiagnosis(dicomDiagnosis: DicomDiagnosis): DicomInterpretation {
            val o = viewState.dicomInterpretation
            o.setDicomDiagnosis(dicomDiagnosis)
            return o
        }
    }

    inner class EditingToolCheckBoxReducer(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent)
        : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {
            Log.w(TAG, "EditingToolReducer: ${viewEvent.javaClass.name}")
            when(viewEvent) {
                is InterpretationViewEvent.OnToolClicked -> {
                    viewModel.setToolUsing(viewEvent.interpretationViewTool)

                    if (viewEvent.interpretationViewTool.imageView?.id == R.id.bt_tool_diagnosis) {
                        Log.w(TAG, "Clicked into R.id.bt_tool_diagnosis")
                    }
                    when (viewEvent.interpretationViewTool) {


                        is InterpretationViewTool.OnClickDiagnosis -> {

                            viewModel.viewStates().value?.let {
                                Log.w(TAG, "OnClickDiagnosis-dicomDiagnosis: ${it.dicomDiagnosis}")
//                                Log.w(TAG, "OnClickDiagnosis-dicomDiagnosis: ${it.}")

                                var tool = viewEvent.interpretationViewTool.copy(dicomDiagnosis = it.dicomDiagnosis)
                                return InterpretationViewObject(viewState = null,
                                    viewEffect = InterpretationViewEffect.RenderToolUsing(viewEvent.copy(interpretationViewTool = tool))
                                )
                            }
                        }
                    }
                }

                is InterpretationViewEvent.OnCheckBoxClicked -> {
                    Log.w(TAG, "InterpretationViewEvent.OnCheckBoxClicked")
                    val interpretationViewTool = viewEvent.interpretationViewTool
                    when (interpretationViewTool) {
                        is InterpretationViewTool.OnClickCheckBox -> {
                            val cb = interpretationViewTool.checkBox

                            // checkbox is perimeter (gls draw)
                            when (cb.id) {
                                R.id.cb_perimeter -> {
                                    Log.w(TAG, "Click into cb_perimeter: ${cb.isChecked}")
                                    viewModel.setIsGls(cb.isChecked)
                                }

                                R.id.cb_esv -> {
                                    Log.w(TAG, "Click into cb_esv: ${cb.isChecked} frameIndex: ${viewModel.getCurrentFrameIndex()}")
                                    viewModel.viewStates().value?.let {
                                        return InterpretationViewObject(
                                            viewState = it.copy(dicomInterpretation = setESVToViewState(cb.isChecked), status = InterpretationViewStatus.OnComponentClick),
                                            viewEffect = viewModel.getRenderFragmentViewEffect()
                                        )
                                    }
                                }

                                R.id.cb_edv -> {
                                    Log.w(TAG, "Click into cb_edv: ${cb.isChecked} frameIndex: ${viewModel.getCurrentFrameIndex()}")
                                    viewModel.viewStates().value?.let {
                                        return InterpretationViewObject(
                                            viewState = it.copy(dicomInterpretation = setEDVToViewState(cb.isChecked), status = InterpretationViewStatus.OnComponentClick),
                                            viewEffect = viewModel.getRenderFragmentViewEffect()
                                        )
                                    }
                                }
                            }

                        }

                    }
                }
            }
            return InterpretationViewObject(viewState = null, viewEffect = InterpretationViewEffect.RenderToolUsing(viewEvent))
        }

        fun setEDVToViewState(isEDV: Boolean): DicomInterpretation {
            val o = viewState.dicomInterpretation
            o.setEDV(viewModel.getCurrentFrameIndex(), isEDV=isEDV)
            return o
        }

        fun setESVToViewState(isESV: Boolean): DicomInterpretation {
            val o = viewState.dicomInterpretation
            o.setESV(viewModel.getCurrentFrameIndex(), isESV=isESV)
            return o
        }

    }


    inner class PlaybackReducer(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent)
        : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {
            when(viewEvent) {
                is InterpretationViewEvent.RenderFragmentView -> {
                    viewModel.setCurrentFrameIndexWithClick(viewEvent.buttonID, viewEvent.isLongClicked, viewEvent.isPlayingNextFrame)
                }
            }
            return InterpretationViewObject(viewState = viewState.copy(status = InterpretationViewStatus.OnPlayBackClick), viewEffect = viewModel.getRenderFragmentViewEffect())
        }

    }
}
