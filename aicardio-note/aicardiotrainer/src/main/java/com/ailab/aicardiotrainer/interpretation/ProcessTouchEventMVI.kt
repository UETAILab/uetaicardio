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

import android.util.Log

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

        const val TAG = "ProcessTouchEventMVI"

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
            is InterpretationViewEvent.ProcessTouchEvent -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(TouchEventReducer(interpretationActVM, it, interpretationViewEvent))
                }
            }
        }
    }

    inner class TouchEventReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.ProcessTouchEvent)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            val isValidFrameState = viewModel.getIsValidFrameState()
            if (!isValidFrameState) return StateEffectObject()

            val touchEvent = TouchEventObject(frameIdx = viewModel.getCurrentFrameIndex(), scale = viewEvent.view.getScale(), event = viewEvent.event, ix = viewEvent.ix, iy = viewEvent.iy)
            Log.w(TAG, "TouchEventReducer: ${touchEvent}")
            // check type (point/ boundary/ length/ area)

            val toolName = viewModel.getToolNameUsing()
            val toolClickedType = viewModel.getToolClickedType()

            when (toolName) {
                InterpretationActVM.TOOL_DRAW_POINT -> {
                    when (toolClickedType) {
                        true -> {
                            // modify point
                            ProcessAddAndModifyMVI.process(viewModel, InterpretationViewEvent.ModifyDrawPoint(obj=touchEvent.copy(key=viewModel.keyPoint, modifyPointIdx = viewModel.getModifyPointIndex() ) ) )
                        }
                        false -> {
                            // add point
                            ProcessAddAndModifyMVI.process(viewModel, InterpretationViewEvent.DrawPoint( obj=touchEvent.copy(key=viewModel.keyPoint) ))
                        }
                    }
                }

                InterpretationActVM.TOOL_DRAW_BOUNDARY -> {
                    when (toolClickedType) {
                        true -> {
                            ProcessAddAndModifyMVI.process(viewModel, InterpretationViewEvent.ModifyDrawBoundary(obj=touchEvent.copy(key = viewModel.keyBoundary, modifyBoundaryIndex = viewModel.getModifyBoundaryIndex())) )
                        }
                        false -> {
                            ProcessAddAndModifyMVI.process(viewModel, InterpretationViewEvent.DrawBoundary( obj=touchEvent.copy(key=viewModel.keyBoundary) ))
                        }
                    }
                }

                InterpretationActVM.TOOL_MEASURE_LENGTH -> {
                    if (!toolClickedType) {
                        ProcessAddAndModifyMVI.process(viewModel, InterpretationViewEvent.DrawPoint( obj=touchEvent.copy(key= DicomAnnotation.MEASURE_LENGTH) ) )
                    }

                }

                InterpretationActVM.TOOL_MEASURE_AREA -> {
                    if (!toolClickedType) {
                        ProcessAddAndModifyMVI.process(viewModel, InterpretationViewEvent.DrawBoundary(obj=touchEvent.copy(key = DicomAnnotation.MEASURE_AREA)) )
                    }
                }

            }

            return StateEffectObject()
            }


        }
}