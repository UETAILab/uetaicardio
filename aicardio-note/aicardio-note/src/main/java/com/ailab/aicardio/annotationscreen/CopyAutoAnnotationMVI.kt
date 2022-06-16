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

import android.widget.Toast
import com.ailab.aicardio.repository.AnnotationStateEffectObject
import kotlinx.android.synthetic.main.activity_annotate.*

class CopyAutoAnnotationMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: CopyAutoAnnotationMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: CopyAutoAnnotationMVI()
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
        when (viewEffect) {
            is AnnotationViewEffect.CopyAutoAllFrames -> {
                Toast.makeText(annotationActivity, viewEffect.message, Toast.LENGTH_SHORT).show()
            }
            is AnnotationViewEffect.CopyAutoOneFrame -> {
                Toast.makeText(annotationActivity, viewEffect.message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
        when (viewState.status) {
            is AnnotationViewStatus.CopyAutoAllFrames -> {
                annotationActivity.iv_draw_canvas.invalidate()
            }
            is AnnotationViewStatus.CopyAutoOneFrame -> {
                annotationActivity.iv_draw_canvas.invalidate()

            }
        }
    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
        annotationActVM.viewStates().value?.let {
            annotationActVM.reduce(Reducer(annotationActVM, it, annotationViewEvent))
        }
    }

    inner class Reducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {

            when (viewEvent) {

                AnnotationViewEvent.CopyAutoOneFrame -> {
                    val o = viewState.dicomAnnotation
                    o.setIsCopy(viewModel.getCurrentFrameIndex(), true)
                    o.copyMachineAnnotation(viewModel.getCurrentFrameIndex(), viewState.machineAnnotation)

                    val new_viewState = viewState.copy(dicomAnnotation = o, status = AnnotationViewStatus.CopyAutoOneFrame)
                    val new_viewEffect = AnnotationViewEffect.CopyAutoOneFrame(message = "CopyAutoOneFrame")
                    return AnnotationStateEffectObject(new_viewState, new_viewEffect)
                }

                AnnotationViewEvent.CopyAutoAllFrames -> {
                    val o = viewState.dicomAnnotation
                    o.setAllFrameIsCopy(true)
                    o.copyAllFrameMachineAnnotation(viewState.machineAnnotation)

                    val new_viewState = viewState.copy(dicomAnnotation = o, status = AnnotationViewStatus.CopyAutoAllFrames)
                    val new_viewEffect = AnnotationViewEffect.CopyAutoAllFrames(message = "CopyAutoAllFrames")
                    return AnnotationStateEffectObject(new_viewState, new_viewEffect)

                }
                else -> return AnnotationStateEffectObject()
            }

        }

    }

}