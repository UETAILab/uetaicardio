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
import android.widget.Toast
import androidx.lifecycle.viewModelScope
import com.ailab.aicardio.repository.AnnotationStateEffectObject
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class UniDirectionMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: UniDirectionMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: UniDirectionMVI()
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
            is AnnotationViewEffect.Test -> {
                Toast.makeText(annotationActivity, viewEffect.message, Toast.LENGTH_SHORT).show()
            }
            is AnnotationViewEffect.TestAsync -> {
                Toast.makeText(annotationActivity, viewEffect.message, Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun renderViewState(annotationActivity: AnnotationActivity, viewState: AnnotationViewState) {
        when (viewState.status) {
            is AnnotationViewStatus.Test -> {
                val newText = "-- ${viewState.status.message} --"
//                annotationActivity.bt_test.text = newText
            }
            is AnnotationViewStatus.TestAsync -> {
                val newText = "++ ${viewState.status.message} ++"
//                annotationActivity.bt_test.text = newText
            }
        }
    }

    fun process(annotationActVM: AnnotationActVM, annotationViewEvent: AnnotationViewEvent) {
        if (annotationViewEvent is AnnotationViewEvent.Test) {
            annotationActVM.viewStates().value?.let {
                annotationActVM.reduce(Reducer(annotationActVM, it, annotationViewEvent))
            }
        }
    }

    inner class Reducer(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.Test)
        : AnnotationActReducer(viewModel, viewState, viewEvent) {

        suspend fun asyncRun() : Int = withContext(Dispatchers.IO) {
            repeat(100000000) {
                100*100
            }
            return@withContext 1234
        }

        override fun reduce(): AnnotationStateEffectObject {
            val viewState = viewState.copy(status = AnnotationViewStatus.Test(message = viewEvent.message))
            val viewEffect = AnnotationViewEffect.Test(message = viewEvent.message)

            viewModel.viewModelScope.launch {
                Log.w("PlaybackReducer", "start async run")
                val result = asyncRun()
                Log.w("PlaybackReducer", "end async run")

                viewModel.viewStates().value?.let {
                    viewModel.reduce(ReducerAsync(viewModel, it, AnnotationViewEvent.TestAsync(message = "Test Async $result")))
                }
            }

            return AnnotationStateEffectObject(viewState, viewEffect)
        }

    }

    inner class ReducerAsync(viewModel: AnnotationActVM, viewState: AnnotationViewState, val viewEvent: AnnotationViewEvent.TestAsync) : AnnotationActReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): AnnotationStateEffectObject {
            Log.w("ReducerAsync", "start reducer async")
            val viewState = viewState.copy(status = AnnotationViewStatus.TestAsync(message = viewEvent.message))
            val viewEffect = AnnotationViewEffect.TestAsync(message = viewEvent.message)
            Log.w("ReducerAsync", "end reducer async")
            return AnnotationStateEffectObject(viewState, viewEffect)
        }

    }
}