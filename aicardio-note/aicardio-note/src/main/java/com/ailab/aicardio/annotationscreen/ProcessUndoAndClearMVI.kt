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

import com.ailab.aicardio.R
import com.ailab.aicardio.repository.AnnotationStateEffectObject
import com.ailab.aicardio.repository.DicomAnnotation

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

        fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
            getInstance().process(annotationActVM, annotationViewEvent)
        }

        fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
            getInstance().renderViewState(annotationActivity, viewState)
        }

        fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
            getInstance().renderViewEffect(annotationActivity, viewEffect)
        }
    }

    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {}

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {}

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
        when (annotationViewEvent) {
            is AnnotationViewEvent.UndoAnnotation -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(UndoAnnotationReducer(annotationActVM, it, annotationViewEvent))
                }
            }

            is AnnotationViewEvent.ClearAnnotation -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(ClearAnnotationReducer(annotationActVM, it, annotationViewEvent))
                }
            }
        }
    }

    inner class UndoAnnotationReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.UndoAnnotation)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {

            if (viewModel.getIsValidFrameState()) {
                viewModel.viewStates().value?.let {
                    when (viewModel.getCurrentTool().first) {
                        R.id.bt_draw_point -> {
                            // no check toolClickedType
                            viewModel.reduce(UndoPointReducer(viewModel, it, AnnotationViewEvent.UndoPoint(key=viewModel.keyPoint, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        R.id.bt_draw_boundary -> {
                            viewModel.reduce(UndoBoundaryReducer(viewModel, it, AnnotationViewEvent.UndoBoundary(key=viewModel.keyBoundary, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        R.id.bt_measure_length -> {
                            viewModel.reduce(UndoPointReducer(viewModel, it, AnnotationViewEvent.UndoPoint(key= DicomAnnotation.MEASURE_LENGTH, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        R.id.bt_measure_area -> {
                            viewModel.reduce(UndoBoundaryReducer(viewModel, it, AnnotationViewEvent.UndoBoundary(key= DicomAnnotation.MEASURE_AREA, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                    }
                }

            }
            return AnnotationStateEffectObject()
        }

    }

    inner class ClearAnnotationReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.ClearAnnotation)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {

            if (viewModel.getIsValidFrameState()) {
                viewModel.viewStates().value?.let {
                    when (viewModel.getCurrentTool().first) {
                        R.id.bt_draw_point -> {
                            viewModel.reduce(ClearPointReducer(viewModel, it, AnnotationViewEvent.ClearPoint(key=viewModel.keyPoint, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        R.id.bt_draw_boundary -> {
                            viewModel.reduce(ClearBoundaryReducer(viewModel, it, AnnotationViewEvent.ClearBoundary(key=viewModel.keyBoundary, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        R.id.bt_measure_length -> {
                            viewModel.reduce(ClearPointReducer(viewModel, it, AnnotationViewEvent.ClearPoint(key= DicomAnnotation.MEASURE_LENGTH, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                        R.id.bt_measure_area -> {
                            viewModel.reduce(ClearBoundaryReducer(viewModel, it, AnnotationViewEvent.ClearBoundary(key= DicomAnnotation.MEASURE_AREA, frameIdx=viewModel.getCurrentFrameIndex())))
                        }

                    }
                }

            }

            return AnnotationStateEffectObject()
        }

    }


    inner class UndoPointReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.UndoPoint)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {
            val o = viewState.dicomAnnotation
            o.removePoint(frameIdx = viewEvent.frameIdx, key = viewEvent.key)
            o.changeLength(frameIdx = viewEvent.frameIdx, key = viewEvent.key, tags = viewState.tagsDicom)
            return AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.UndoPoint),
                viewModel.getRenderAnnotationFrame()
            )
        }

    }

    inner class ClearPointReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.ClearPoint)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {
            val o = viewState.dicomAnnotation
            o.clearPoints(frameIdx = viewEvent.frameIdx, key = viewEvent.key)
            o.changeLength(frameIdx = viewEvent.frameIdx, key = viewEvent.key, tags = viewState.tagsDicom)
            return AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.ClearPoint),
                viewModel.getRenderAnnotationFrame()
            )
        }

    }

    inner class UndoBoundaryReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.UndoBoundary)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {
            val o = viewState.dicomAnnotation
            o.removePath(frameIdx = viewEvent.frameIdx, key = viewEvent.key)
            o.changeArea(frameIdx=viewEvent.frameIdx, key=viewEvent.key, tags=viewState.tagsDicom )

            return AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.UndoBoundary),
                viewModel.getRenderAnnotationFrame()
            )
        }

    }

    inner class ClearBoundaryReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.ClearBoundary)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {
            val o = viewState.dicomAnnotation
            o.clearBoundary(frameIdx = viewEvent.frameIdx, key = viewEvent.key)
            o.changeArea(frameIdx=viewEvent.frameIdx, key=viewEvent.key, tags=viewState.tagsDicom )

            return AnnotationStateEffectObject(
                viewState.copy(status = AnnotationViewStatus.ClearBoundary),
                viewModel.getRenderAnnotationFrame()
            )
        }

    }


}