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
import com.ailab.aicardiotrainer.R
import kotlinx.android.synthetic.main.activity_interpretation.*

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

class SetToolUsingMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: SetToolUsingMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: SetToolUsingMVI()
                        .also { instance = it }
            }

        const val TAG = "OnToolUsingMVI"

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
        when (viewEffect) {
            is InterpretationViewEffect.RenderButtonTool -> {
                renderButtonTool(interpretationActivity, viewEvent = viewEffect.viewEvent)
            }
            is InterpretationViewEffect.RenderClearCurrentTool -> {
                renderClearButtonTool(interpretationActivity, viewEvent = viewEffect.viewEvent)
            }
        }
    }

    private fun renderViewState(interpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {

    }

    fun process(interpretationActVM: InterpretationActVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {
            is InterpretationViewEvent.OnToolUsing -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(SetToolUsingReducer(interpretationActVM, it, interpretationViewEvent))
                }
            }
            is InterpretationViewEvent.ClearCurrentTool -> {
                interpretationActVM.setToolNameUsing(InterpretationActVM.TOOL_NO_ACTION, false)
//                annotationActVM.setCurrentToolId(null)
                interpretationActVM.reduce(StateEffectObject(null, viewEffect = InterpretationViewEffect.RenderClearCurrentTool(interpretationViewEvent)))
            }
        }
    }


    inner class SetToolUsingReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.OnToolUsing)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {


        override fun reduce(): StateEffectObject {
            val toolName = viewEvent.toolName
            val toolTypeClick = viewEvent.toolTypeClick
            Log.w(TAG, "toolName ${toolName} toolTypeClick ${toolTypeClick}")
            when (toolName) {
                InterpretationActVM.TOOL_CHECK_BOX_PERIMETER -> {
                    viewModel.setIsGls(viewEvent.toolTypeClick)
                    SetToolUsingMVI.process(viewModel, InterpretationViewEvent.ClearCurrentTool)
                }

                InterpretationActVM.TOOL_CHECK_BOX_ESV -> {
                    if (!viewEvent.isPlaying) {
                        return StateEffectObject(viewState.copy(
                            dicomAnnotation = setESVToViewState(isESV = toolTypeClick),
                            status = InterpretationViewStatus.SetESVState
                        ), viewModel.getRenderReadingMediaFrame())
                    }
                }

                InterpretationActVM.TOOL_CHECK_BOX_EDV -> {
                    if (!viewEvent.isPlaying) {
                        return StateEffectObject(viewState.copy(
                            dicomAnnotation = setEDVToViewState(isEDV = toolTypeClick),
                            status = InterpretationViewStatus.SetEDVState
                        ), viewModel.getRenderReadingMediaFrame())
                    }
                }

                else -> {
                    viewModel.setToolNameUsing(toolName, toolTypeClick)
                }
            }


            return StateEffectObject(viewEffect = InterpretationViewEffect.RenderButtonTool(viewEvent))
        }

        fun setEDVToViewState(isEDV: Boolean): DicomAnnotation {
            val o = viewState.dicomAnnotation
            o.setEDV(viewModel.getCurrentFrameIndex(), isEDV=isEDV)
            return o
        }

        fun setESVToViewState(isESV: Boolean): DicomAnnotation {
            val o = viewState.dicomAnnotation
            o.setESV(viewModel.getCurrentFrameIndex(), isESV=isESV)
            return o
        }

    }
    fun renderClearButtonTool(interpretationActivity: InterpretationActivity, viewEvent: InterpretationViewEvent.ClearCurrentTool) {
        interpretationActivity.bt_tool_interpretation.setImageResource(R.drawable.ic_no_action)
        interpretationActivity.bt_tool_interpretation.background = interpretationActivity.getDrawable(R.drawable.rounded_rectangle)
    }

    fun renderButtonTool(interpretationActivity: InterpretationActivity, viewEvent: InterpretationViewEvent.OnToolUsing) {
        val toolName = viewEvent.toolName
        val toolTypeClick = viewEvent.toolTypeClick

        when (toolName) {
            InterpretationActVM.TOOL_DRAW_POINT -> {
                interpretationActivity.bt_tool_interpretation.setImageResource(R.drawable.ic_draw_point)
            }
            InterpretationActVM.TOOL_DRAW_BOUNDARY -> {
                interpretationActivity.bt_tool_interpretation.setImageResource(R.drawable.ic_draw_boundary)
            }

            InterpretationActVM.TOOL_MEASURE_LENGTH -> {
                interpretationActivity.bt_tool_interpretation.setImageResource(R.drawable.ic_measure_length)

            }

            InterpretationActVM.TOOL_MEASURE_AREA -> {
                interpretationActivity.bt_tool_interpretation.setImageResource(R.drawable.ic_measure_area)

            }
            else -> interpretationActivity.bt_tool_interpretation.setImageResource(R.drawable.ic_no_action)
        }

        if (toolTypeClick) interpretationActivity.bt_tool_interpretation.background = interpretationActivity.getDrawable(R.drawable.rounded_rectangle_long_clicked)
        else interpretationActivity.bt_tool_interpretation.background = interpretationActivity.getDrawable(R.drawable.rounded_rectangle)

        interpretationActivity.closeEditingToolDialog()
    }

}