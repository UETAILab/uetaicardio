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
import kotlinx.android.synthetic.main.activity_interpretation.*
import kotlinx.android.synthetic.main.activity_study.*
import kotlinx.coroutines.launch

class StudyRepresentationMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: StudyRepresentationMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: StudyRepresentationMVI()
                        .also { instance = it }
            }

        const val TAG = "StudyRepresentationMVI"

        fun process(InterpretationActVM: InterpretationActVM, InterpretationViewEvent: InterpretationViewEvent) {
            getInstance().process(InterpretationActVM, InterpretationViewEvent)
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
            InterpretationViewStatus.OnDoneLoadingRepresentationStudyInstanceUID -> {
                Log.w(TAG, "renderViewState ${viewState.studyFiles}")
                annotationFragment.studyRepresentationGVAdapter?.submitList(viewState.getListSopInstanceUIDItem())

            }
        }
    }

    fun process(interpretationActVM: InterpretationActVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {

            is InterpretationViewEvent.LoadingRepresentationStudyInstanceUID -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(LoadingStudyRepresentationReducerAsync(interpretationActVM, it, interpretationViewEvent))
                }
            }

        }
    }

    inner class Reducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent)
        : InterpretationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): StateEffectObject {
            when(viewEvent) {

            }
            return StateEffectObject()
        }

    }

    inner class LoadingStudyRepresentationReducerAsync(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent.LoadingRepresentationStudyInstanceUID) : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            if (viewState.status is InterpretationViewStatus.OnLoadingRepresentationStudyInstanceUID) return StateEffectObject()
            val result = StateEffectObject(viewState = viewState.copy(status = InterpretationViewStatus.OnLoadingRepresentationStudyInstanceUID))

            viewModel.viewModelScope.launch {

                val resultLaunch = diskRepository.getRepresentationInStudyInstanceUID(viewEvent.studyInstanceUID)
                if (resultLaunch.error == false) {
                    // process
                    Log.w(TAG, "No error: ${resultLaunch.data.representationFiles}")
                    viewModel.reduce(
                        StateEffectObject(
                            viewState = viewState.copy(status = InterpretationViewStatus.OnDoneLoadingRepresentationStudyInstanceUID, studyFiles = resultLaunch.data.representationFiles, studyRepresentation = resultLaunch.data.representationBitmap),
                            viewEffect = InterpretationViewEffect.ShowToast("DONE Loading Representation StudyInstanceUID")
                        )
                    )
                } else {
                    // error when loading presentatinon
                    viewModel.reduce(
                        StateEffectObject(
                            null,
                            viewEffect = InterpretationViewEffect.ShowToast("ERROR Loading Representation StudyInstanceUID")
                        )
                    )
                }

            }

            return result
        }

    }
}