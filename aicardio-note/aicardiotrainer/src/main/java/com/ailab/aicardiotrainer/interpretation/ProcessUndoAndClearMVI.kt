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

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.android.synthetic.main.activity_interpretation.*
import kotlinx.coroutines.launch

class ProcessUndoAndClearMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: ProcessUndoAndClearMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: ProcessUndoAndClearMVI()
                        .also { instance = it }
            }

        const val TAG = "ProcessUndoAndClearMVI"

        fun process(interpretationActVM: InterpretationActVM, InterpretationViewEvent: InterpretationViewEvent) {
            getInstance().process(interpretationActVM, InterpretationViewEvent)
        }

        fun renderViewState(interpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {
            getInstance().renderViewState(interpretationActivity, viewState)
        }

        fun renderViewEffect(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect) {
            getInstance().renderViewEffect(interpretationActivity, viewEffect)
        }
    }


    private fun renderViewEffect(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect) {

    }

    private fun renderViewState(interpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {

    }

    fun process(interpretationActVM: InterpretationActVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {
            is InterpretationViewEvent.UndoAnnotation -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(UndoAnnotationReducer(interpretationActVM, it, interpretationViewEvent))
                }
            }

            is InterpretationViewEvent.ClearAnnotation -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(ClearAnnotationReducer(interpretationActVM, it, interpretationViewEvent))
                }
            }
        }
    }
    inner class UndoAnnotationReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.UndoAnnotation)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            if (viewModel.getIsValidFrameState()) {
                viewModel.viewStates().value?.let {
                    when (viewModel.getToolNameUsing()) {
                        InterpretationActVM.TOOL_DRAW_POINT -> {
                            // no check toolClickedType
                            viewModel.reduce(UndoPointReducer(viewModel, it, InterpretationViewEvent.UndoPoint(key=viewModel.keyPoint, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        InterpretationActVM.TOOL_DRAW_BOUNDARY -> {
                            viewModel.reduce(UndoBoundaryReducer(viewModel, it, InterpretationViewEvent.UndoBoundary(key=viewModel.keyBoundary, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        InterpretationActVM.TOOL_MEASURE_LENGTH -> {
                            viewModel.reduce(UndoPointReducer(viewModel, it, InterpretationViewEvent.UndoPoint(key= DicomAnnotation.MEASURE_LENGTH, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        InterpretationActVM.TOOL_MEASURE_AREA -> {
                            viewModel.reduce(UndoBoundaryReducer(viewModel, it, InterpretationViewEvent.UndoBoundary(key= DicomAnnotation.MEASURE_AREA, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                    }
                }

            }
            return StateEffectObject()
        }

    }

    inner class ClearAnnotationReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.ClearAnnotation)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            if (viewModel.getIsValidFrameState()) {
                viewModel.viewStates().value?.let {
                    when (viewModel.getToolNameUsing()) {

                        InterpretationActVM.TOOL_DRAW_POINT -> {
                            viewModel.reduce(ClearPointReducer(viewModel, it, InterpretationViewEvent.ClearPoint(key=viewModel.keyPoint, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        InterpretationActVM.TOOL_DRAW_BOUNDARY -> {
                            viewModel.reduce(ClearBoundaryReducer(viewModel, it, InterpretationViewEvent.ClearBoundary(key=viewModel.keyBoundary, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        InterpretationActVM.TOOL_MEASURE_LENGTH -> {
                            viewModel.reduce(ClearPointReducer(viewModel, it, InterpretationViewEvent.ClearPoint(key= DicomAnnotation.MEASURE_LENGTH, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        InterpretationActVM.TOOL_MEASURE_AREA -> {
                            viewModel.reduce(ClearBoundaryReducer(viewModel, it, InterpretationViewEvent.ClearBoundary(key= DicomAnnotation.MEASURE_AREA, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                    }
                }

            }

            return StateEffectObject()
        }

    }


    inner class UndoPointReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.UndoPoint)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {
            val o = viewState.dicomAnnotation
            o.removePoint(frameIdx = viewEvent.frameIdx, key = viewEvent.key)
            o.changeLength(frameIdx = viewEvent.frameIdx, key = viewEvent.key, tags = viewState.tagsDicom)
            return StateEffectObject(
                viewState.copy(status = InterpretationViewStatus.UndoPoint),
                viewModel.getRenderReadingMediaFrame()
            )
        }

    }

    inner class ClearPointReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.ClearPoint)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {
            val o = viewState.dicomAnnotation
            o.clearPoints(frameIdx = viewEvent.frameIdx, key = viewEvent.key)
            o.changeLength(frameIdx = viewEvent.frameIdx, key = viewEvent.key, tags = viewState.tagsDicom)
            return StateEffectObject(
                viewState.copy(status = InterpretationViewStatus.ClearPoint),
                viewModel.getRenderReadingMediaFrame()
            )
        }

    }

    inner class UndoBoundaryReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.UndoBoundary)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {
            val o = viewState.dicomAnnotation
            o.removePath(frameIdx = viewEvent.frameIdx, key = viewEvent.key)
            o.changeArea(frameIdx=viewEvent.frameIdx, key=viewEvent.key, tags=viewState.tagsDicom )

            return StateEffectObject(
                viewState.copy(status = InterpretationViewStatus.UndoBoundary),
                viewModel.getRenderReadingMediaFrame()
            )
        }

    }

    inner class ClearBoundaryReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.ClearBoundary)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {
            val o = viewState.dicomAnnotation
            o.clearBoundary(frameIdx = viewEvent.frameIdx, key = viewEvent.key)
            o.changeArea(frameIdx=viewEvent.frameIdx, key=viewEvent.key, tags=viewState.tagsDicom )

            return StateEffectObject(
                viewState.copy(status = InterpretationViewStatus.ClearBoundary),
                viewModel.getRenderReadingMediaFrame()
            )
        }

    }

}