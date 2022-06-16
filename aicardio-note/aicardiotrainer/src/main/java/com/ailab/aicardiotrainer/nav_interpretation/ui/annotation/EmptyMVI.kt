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

class EmptyMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: EmptyMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: EmptyMVI()
                        .also { instance = it }
            }

        const val TAG = "EmptyMVI"

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

    private fun renderViewEffect(annotationFragment: AnnotationFragment, viewEffect: InterpretationViewEffect) {
    }

    private fun renderViewState(annotationFragment: AnnotationFragment, viewState: InterpretationViewState) {
    }

    fun process(interpretationActVM: InterpretationActVM, InterpretationViewEvent: InterpretationViewEvent) {

        when (InterpretationViewEvent) {

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

    inner class ReducerAsync(viewModel: InterpretationActVM, viewState: InterpretationViewState, viewEvent: InterpretationViewEvent) : InterpretationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): StateEffectObject {

            return StateEffectObject()
        }

    }
}