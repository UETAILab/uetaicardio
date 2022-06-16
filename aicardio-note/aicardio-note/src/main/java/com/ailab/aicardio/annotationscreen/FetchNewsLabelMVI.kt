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
import com.ailab.aicardio.repository.AnnotationRepository
import com.ailab.aicardio.repository.AnnotationStateEffectObject

class FetchNewsLabelMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: FetchNewsLabelMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: FetchNewsLabelMVI()
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
        private val annotationRepository = AnnotationRepository.getInstance()

    }

    private fun renderViewEffect(annotationActivity: AnnotationActivity, viewEffect: AnnotationViewEffect) {
    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {

    }

    inner class Reducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.Test)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): AnnotationStateEffectObject {
            return AnnotationStateEffectObject()
        }

    }

    inner class ReducerAsync(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.TestAsync) : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {
//            Log.w("ReducerAsync", "start reducer async")
////            val viewState = viewState.copy(status = AnnotationViewStatus.TestAsync(message = viewEvent.message))
////            val viewEffect = AnnotationViewEffect.TestAsync(message = viewEvent.message)
//            Log.w("ReducerAsync", "end reducer async")
//
//            annotationRepository.getAnnotationFromFile()

            return AnnotationStateEffectObject()
        }

    }
}