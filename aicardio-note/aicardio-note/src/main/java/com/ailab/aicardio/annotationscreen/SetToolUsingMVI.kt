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

import android.util.Log
import com.ailab.aicardio.R
import com.ailab.aicardio.TAG_SHORT_CLICKED
import com.ailab.aicardio.repository.AnnotationStateEffectObject
import com.ailab.aicardio.repository.DicomAnnotation
import com.ailab.aicardio.repository.DicomDiagnosis
import kotlinx.android.synthetic.main.activity_annotate.*

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

        const val TAG = "SetToolUsingUniDirectionMVI"

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
        when (viewEffect) {
            is AnnotationViewEffect.RenderButtonTool -> {

                renderButtonTool(annotationActivity, viewEffect.button, viewEffect.typeClicked)
            }
            is AnnotationViewEffect.ShowDiagnosisDialog -> {

                annotationActivity.showDiagnosisDialog(viewEffect.dicomDiagnosis)

            }

            is AnnotationViewEffect.RenderDiagnosisTool -> {
                renderDiagnosisTool(annotationActivity, viewEffect.dicomDiagnosis)
            }
        }
    }

    fun renderDiagnosisTool(annotationActivity: AnnotationActivity, dicomDiagnosis: DicomDiagnosis) {
        annotationActivity.bt_diagnosis.text = dicomDiagnosis.chamber
        annotationActivity.tools[R.id.bt_diagnosis] = Pair(annotationActivity.bt_diagnosis, Pair(dicomDiagnosis.chamber, dicomDiagnosis.chamber))
    }


    fun renderButtonTool(annotationActivity: AnnotationActivity, button: Int, typeClicked: String) {
        Log.w(AnnotationActivity.TAG, "$button, $typeClicked")

        when(button) {
            R.id.bt_draw_point -> {
                annotationActivity.iv_draw_canvas.isZooming = false
            }
            R.id.bt_draw_boundary -> {
                annotationActivity.iv_draw_canvas.isZooming = false
            }

            R.id.bt_measure_length -> {
                annotationActivity.iv_draw_canvas.isZooming = false
            }
            R.id.bt_measure_area -> {
                annotationActivity.iv_draw_canvas.isZooming = false
            }


            R.id.bt_zooming_and_drag -> {
                annotationActivity.iv_draw_canvas.isZooming = !annotationActivity.iv_draw_canvas.isZooming

                annotationActivity.bt_zooming_and_drag.background = if (annotationActivity.iv_draw_canvas.isZooming) annotationActivity.getDrawable(R.drawable.custom_button_enabled_color)
                 else annotationActivity.getDrawable(R.drawable.custom_button_disabled_color)
            }
        }

        annotationActivity.tools.forEach {
//            Log.w(AnnotationActivity.TAG, "${it.key}, ${it.value.second.first} color: ${it.key == button}")
            if (it.key == button) {
                it.value.first.background = annotationActivity.getDrawable(R.drawable.custom_button_enabled_color)
                it.value.first.text = if(typeClicked == TAG_SHORT_CLICKED) it.value.second.first else it.value.second.second
            } else {
                it.value.first.background = annotationActivity.getDrawable(R.drawable.custom_button_disabled_color)
                it.value.first.text = it.value.second.first

            }
        }
    }


    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {

    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
//        if (annotationViewEvent is AnnotationViewEvent.OnToolUsing) {
//            annotationActVM.viewStates().value?.let {
//                annotationActVM.reduce(SetToolUsingReducer(annotationActVM, it, annotationViewEvent))
//            }
//        }

        when (annotationViewEvent) {
            is AnnotationViewEvent.OnToolUsing -> {
                annotationActVM.viewStates().value?.let {
                    annotationActVM.reduce(SetToolUsingReducer(annotationActVM, it, annotationViewEvent))
                }
            }

            AnnotationViewEvent.ClearCurrentTool -> {
                annotationActVM.setCurrentToolId(null)
                annotationActVM.reduceStateEffectObject(null, AnnotationViewEffect.RenderButtonTool(-1, typeClicked = ""))
            }
        }
    }

    inner class SetToolUsingReducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.OnToolUsing)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {


        override fun reduce(): AnnotationStateEffectObject {

            val toolId = viewEvent.toolId
            val typeClicked = viewEvent.typeClicked
            val isPlaying = viewEvent.isPlaying

            when (toolId) {
                R.id.cb_perimeter -> {
                    // checkbox: pass the typeClicked argument is state of checkbox button
//                Log.w(AnnotationActVM.TAG, "$typeClicked")
                    viewModel.setIsGls(typeClicked.toBoolean())
                    viewModel.process(AnnotationViewEvent.ClearCurrentTool)
                    SetToolUsingMVI.process(viewModel, AnnotationViewEvent.ClearCurrentTool)
                }
                R.id.cb_esv -> {
                    // only act when isPlaying = False
                    Log.w(AnnotationActVM.TAG, "CB_ESV, $typeClicked")
                    val isESV = typeClicked.toBoolean()
                    if (!isPlaying)
                        return AnnotationStateEffectObject(viewState.copy(
                            dicomAnnotation = setESVToViewState(isESV = isESV),
                            status = AnnotationViewStatus.SetESVState
                        ), viewModel.getRenderAnnotationFrame())

                }

                R.id.cb_edv -> {
                    // only act when isPlaying = False
                    val isEDV = typeClicked.toBoolean()
                    if (!isPlaying)
                        return AnnotationStateEffectObject(viewState.copy(
                            dicomAnnotation = setEDVToViewState(isEDV = isEDV),
                            status = AnnotationViewStatus.SetEDVState
                        ), viewModel.getRenderAnnotationFrame())
                }

                // label tool - tqlong
                R.id.bt_diagnosis -> {
                    return AnnotationStateEffectObject(null,
                        AnnotationViewEffect.ShowDiagnosisDialog(viewState.dicomDiagnosis)
                    )
                }

                else -> {
                    viewModel.setCurrentToolId(toolId)
                    viewModel.setToolClickedType(typeClicked)
                    return AnnotationStateEffectObject(
                        null,
                        AnnotationViewEffect.RenderButtonTool(toolId, typeClicked)
                    )

                }
            }
            return AnnotationStateEffectObject()
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

}