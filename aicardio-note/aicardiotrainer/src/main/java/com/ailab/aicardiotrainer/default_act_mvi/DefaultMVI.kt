/*
 * Copyright 2021 UET-AILAB
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

package com.ailab.aicardiotrainer.default_act_mvi

import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class DefaultMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: DefaultMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: DefaultMVI()
                        .also { instance = it }
            }

        const val TAG = "DefaultMVI"

        fun process(defaultVM: DefaultVM, DefaultEvent: DefaultEvent) {
            getInstance().process(defaultVM, DefaultEvent)
        }

        fun renderViewState(defaultActivity: DefaultActivity, viewState: DefaultState) {
            getInstance().renderViewState(defaultActivity, viewState)
        }

        fun renderViewEffect(defaultActivity: DefaultActivity, viewEffect: DefaultEffect) {
            getInstance().renderViewEffect(defaultActivity, viewEffect)
        }
    }

    private fun renderViewEffect(defaultActivity: DefaultActivity, viewEffect: DefaultEffect) {

    }

    private fun renderViewState(defaultActivity: DefaultActivity, viewState: DefaultState) {

    }

    fun process(defaultVM: DefaultVM, defaultEvent: DefaultEvent) {

        when (defaultEvent) {

        }
    }

    inner class Reducer(viewModel: DefaultVM, viewState: DefaultState, val viewEvent: DefaultEvent)
        : DefaultReducer(viewModel, viewState, viewEvent) {
        override fun reduce(): DefaultObject {
            when(viewEvent) {

            }
            return DefaultObject()
        }

    }

    inner class ReducerAsync(viewModel: DefaultVM, viewState: DefaultState, viewEvent: DefaultEvent) : DefaultReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): DefaultObject {
            viewModel.viewModelScope.launch {
                val result = asyncRun()
                viewModel.viewStates().value?.let {
                    viewModel.reduce(DefaultObject())
                }
            }
            return DefaultObject()
        }

        suspend fun asyncRun() : Int = withContext(Dispatchers.IO) {
            repeat(100000000) {
                100*100
            }
            return@withContext 1234
        }
    }
}