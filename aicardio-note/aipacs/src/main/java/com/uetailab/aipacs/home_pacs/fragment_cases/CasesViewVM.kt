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

package com.uetailab.aipacs.home_pacs.fragment_cases
import aacmvi.AacMviViewModel
import android.app.Application

class CasesViewVM(application: Application) : AacMviViewModel<CasesViewState, CasesViewEffect, CasesViewEvent>(application) {
    companion object {
        const val TAG = "CasesViewVM"
    }
    var cntClicked = 0
    init {
        viewState = CasesViewState(
            status = CasesViewStatus.Start,
            message = "Start-CasesView-Activity"
        )
    }

    fun getMessage(): String {
        return viewState.message
    }
    fun addCountToMessage() {
        cntClicked += 1
        viewState = viewState.copy(message = "${cntClicked}" )
    }


    fun reduce(reducer: CasesViewReducer) {
        val result = reducer.reduce()
        reduce(result)
    }


    fun reduce(result: CasesViewObject) {
        result.viewState?.let { viewState = it }
        result.viewEffect?.let { viewEffect = it }
    }
}