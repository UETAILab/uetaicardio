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
import androidx.lifecycle.viewModelScope
import kotlinx.android.synthetic.main.fragment_annotation.*
import kotlinx.coroutines.launch

class SopInstanceUIDMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: SopInstanceUIDMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: SopInstanceUIDMVI()
                        .also { instance = it }
            }

        const val TAG = "SopInstanceUIDMVI"
        const val LOAD_NEW_SOPINSTANCEUID = "LOAD_NEW_FILE"

        fun process(interpretationActVM: InterpretationActVM, InterpretationViewEvent: InterpretationViewEvent) {
            getInstance().process(interpretationActVM, InterpretationViewEvent)
        }

        fun renderViewState(annotationFragment: AnnotationFragment, viewState: InterpretationViewState) {
            getInstance().renderViewState(annotationFragment, viewState)
        }

        fun renderViewEffect(annotationFragment: AnnotationFragment, viewEffect: InterpretationViewEffect) {
            getInstance().renderViewEffect(annotationFragment, viewEffect)
        }
    }

    private val diskRepository = DiskRepository.getInstance()


    private fun renderViewEffect(annotationFragment: AnnotationFragment, viewEffect: InterpretationViewEffect) {

    }

    private fun renderViewState(annotationFragment: AnnotationFragment, viewState: InterpretationViewState) {
        when(viewState.status) {
            InterpretationViewStatus.OnDoneLoadingMP4SopInstanceUID -> {
//                interpretationActivity.inter_iv_draw_canvas.setCustomImageBitmap(viewState.sopInstanceUIDBitmaps.get(0))

                annotationFragment.interpretationFrameRVAdapter.submitList(viewState.getSubmitListFrameItem())

                annotationFragment.viewModel.getRenderMP4FrameObject()?.let {
                    InterpretationPlaybackMVI.renderViewEffect(annotationFragment, InterpretationViewEffect.RenderMP4Frame(it))
                }


            }
            InterpretationViewStatus.OnLoadingMP4SopInstanceUID -> {
                annotationFragment.inter_iv_draw_canvas.infoText = LOAD_NEW_SOPINSTANCEUID
                annotationFragment.inter_iv_draw_canvas.setCustomImageBitmap(AnnotationFragment.bitmapHeart)

            }
        }
    }

    fun process(interpretationActVM: InterpretationActVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {
            is InterpretationViewEvent.LoadingMP4SopInstanceUID -> {

                Log.w(TAG, "${interpretationViewEvent.item}")
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(
                        LoadingMP4SopInstanceUIDReducer(
                            interpretationActVM, it, interpretationViewEvent
                        )
                    )
                }


            }
        }
    }

    inner class LoadingMP4SopInstanceUIDReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.LoadingMP4SopInstanceUID)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

//            if (viewState.status is InterpretationViewStatus.OnLoadingMP4SopInstanceUID) return StateEffectObject(viewState = null, viewEffect = InterpretationViewEffect.ShowToast("On Loading MP4SopInstanceUID"))
            if (viewState.status is InterpretationViewStatus.OnLoadingMP4SopInstanceUID) return StateEffectObject()

            val results = StateEffectObject(viewState = viewState.copy(InterpretationViewStatus.OnLoadingMP4SopInstanceUID))

            viewModel.viewModelScope.launch {
                val resultLaunch = diskRepository.getFramesMP4SopInstanceUID(viewEvent.item)
                if (resultLaunch.error == false) {
                    // load dc toan bo cac frame cua video
                    viewModel.viewStates().value?.let {
                        viewModel.reduce(
                            StateEffectObject(
                                viewState = it.copy(status = InterpretationViewStatus.OnDoneLoadingMP4SopInstanceUID, sopInstanceUIDPath = resultLaunch.data.sopIntanceUIDPath, sopInstanceUIDBitmaps = resultLaunch.data.sopInstanceBitmaps)
//                                viewEffect = null, InterpretationViewEffect.ShowToast("Done Loading MP4SopInstanceUID")
                            )
                        )
                    }

                } else {
                    viewModel.viewStates().value?.let {
                        viewModel.reduce(
                            StateEffectObject(
                                viewEffect = InterpretationViewEffect.ShowToast("Loading ERROR MP4SopInstanceUID")
                            )
                        )
                    }

                }
            }

            return results
        }

    }

    inner class ReducerAsync(viewModel: InterpretationActVM, viewState: InterpretationViewState, viewEvent: InterpretationViewEvent) : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            return StateEffectObject()
        }

    }
}