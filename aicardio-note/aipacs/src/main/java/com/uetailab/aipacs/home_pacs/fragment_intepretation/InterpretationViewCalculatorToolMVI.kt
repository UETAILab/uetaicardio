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

package com.uetailab.aipacs.home_pacs.fragment_intepretation

import android.util.Log
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class InterpretationViewCalculatorToolMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: InterpretationViewCalculatorToolMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: InterpretationViewCalculatorToolMVI()
                        .also { instance = it }
            }

        const val TAG = "InterpretationViewCalculatorToolMVI"

        fun process(interpretationViewVM: InterpretationViewVM, InterpretationViewEvent: InterpretationViewEvent) {
            getInstance().process(interpretationViewVM, InterpretationViewEvent)
        }

        fun renderViewState(interpretationViewFragment: InterpretationViewFragment, viewState: InterpretationViewState) {
            getInstance().renderViewState(interpretationViewFragment, viewState)
        }

        fun renderViewEffect(interpretationViewFragment: InterpretationViewFragment, viewEffect: InterpretationViewEffect) {
            getInstance().renderViewEffect(interpretationViewFragment, viewEffect)
        }
    }

    private fun renderViewEffect(interpretationViewFragment: InterpretationViewFragment, viewEffect: InterpretationViewEffect) {

    }

    private fun renderViewState(interpretationViewFragment: InterpretationViewFragment, viewState: InterpretationViewState) {
        when (viewState.status) {
            is InterpretationViewStatus.FetchedData -> {
                val viewEvent = viewState.status.viewEvent
                when(viewEvent) {
                    is InterpretationViewEvent.OnMeasureEFManual -> {
                        interpretationViewFragment.showEFCalculationDialog()
                    }
                }
            }
        }
    }

    fun process(interpretationViewVM: InterpretationViewVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {
            is InterpretationViewEvent.OnMeasureEFManual -> {
//                Log.w(TAG, "InterpretationViewEvent.OnMeasureEFManual")
                interpretationViewVM.viewStates().value?.let {
                    interpretationViewVM.reduce(MeasureEFManualReducerAsync(interpretationViewVM, it, interpretationViewEvent))
                }
            }
        }
    }


    inner class MeasureEFManualReducerAsync(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent) : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {

            viewModel.viewModelScope.launch {
                val result = viewModel.getEFValueFromAllFrame()
//                Log.w(TAG, "MeasureEFManualReducerAsync ${result}")
                viewModel.viewStates().value?.let {
                    viewModel.reduce(InterpretationViewObject(
                        viewState = it.copy(
                            status = InterpretationViewStatus.FetchedData(viewEvent = viewEvent),
                            efObject = result),
                        viewEffect = null
                    ))
                }
            }

            return InterpretationViewObject(viewState=viewState.copy(status = InterpretationViewStatus.OnFetchingData(viewEvent)))
        }

    }
}
