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
import com.ailab.aicardio.TAG_LONG_CLICKED
import com.ailab.aicardio.TAG_SHORT_CLICKED
import com.ailab.aicardio.repository.AnnotationStateEffectObject
import com.ailab.aicardio.repository.DicomAnnotation
import com.ailab.aicardio.repository.TouchEventObject

class ProcessTouchEventMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: ProcessTouchEventMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: ProcessTouchEventMVI()
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

    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {

    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {

    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
        when(annotationViewEvent) {
            is AnnotationViewEvent.ProcessTouchEvent -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(TouchEventReducer(annotationActVM, it, annotationViewEvent))
                }
            }
        }
    }

    inner class TouchEventReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.ProcessTouchEvent)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {

            val isValidFrameState = viewModel.getIsValidFrameState()
            if (!isValidFrameState) return AnnotationStateEffectObject()
            val touchEvent = TouchEventObject(frameIdx = viewModel.getCurrentFrameIndex(), scale = viewEvent.view.getScale(), event = viewEvent.event, ix = viewEvent.ix, iy = viewEvent.iy)

            val currentToolId = viewModel.getCurrentTool()
            val toolClickedType = currentToolId.second

            when (currentToolId.first) {
                null -> {

                }

                R.id.bt_draw_point -> {
                    when (toolClickedType) {
                        TAG_SHORT_CLICKED -> {
                            // Add new point to draw point NOTE: check event.action here
                            ProcessAddAndModifyMVI.process(viewModel, AnnotationViewEvent.DrawPoint( obj=touchEvent.copy(key=viewModel.keyPoint) ))
                        }
                        TAG_LONG_CLICKED -> {
                            ProcessAddAndModifyMVI.process(viewModel, AnnotationViewEvent.ModifyDrawPoint(obj=touchEvent.copy(key=viewModel.keyPoint, modifyPointIdx = viewModel.getModifyPointIndex() ) ) )
                        }
                        else -> {}
                    }

                }
                R.id.bt_draw_boundary -> {
                    when (toolClickedType) {
                        TAG_SHORT_CLICKED -> {
                            ProcessAddAndModifyMVI.process(viewModel, AnnotationViewEvent.DrawBoundary( obj=touchEvent.copy(key=viewModel.keyBoundary) ))
                        }
                        TAG_LONG_CLICKED -> {
                            ProcessAddAndModifyMVI.process(viewModel, AnnotationViewEvent.ModifyDrawBoundary(obj=touchEvent.copy(key = viewModel.keyBoundary, modifyBoundaryIndex = viewModel.getModifyBoundaryIndex())) )
                        }
                        else -> {}
                    }

                }

                R.id.bt_measure_length -> {
                    if (toolClickedType == TAG_SHORT_CLICKED) {
                        ProcessAddAndModifyMVI.process(viewModel, AnnotationViewEvent.DrawPoint( obj=touchEvent.copy(key= DicomAnnotation.MEASURE_LENGTH) ) )
                    }

                }

                R.id.bt_measure_area -> {
                    if (toolClickedType == TAG_SHORT_CLICKED) {
                        ProcessAddAndModifyMVI.process(viewModel, AnnotationViewEvent.DrawBoundary(obj=touchEvent.copy(key = DicomAnnotation.MEASURE_AREA)) )
                    }
                }
            }
            return AnnotationStateEffectObject()
        }

    }

}