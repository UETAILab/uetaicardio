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

package com.uetailab.aipacs.home_pacs

import aacmvi.AacMviViewModel
import android.app.Application
import android.util.Log
import com.uetailab.aipacs.home_pacs.fragment_home.HomeViewVM
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewVM

class HomePacsVM(application: Application) : AacMviViewModel<HomePacsState, HomePacsEffect, HomePacsEvent>(application) {
    companion object {
        const val TAG = "HomePacsVM"
    }
    init {
        viewState = HomePacsState(
            status = HomePacsStatus.Start,
            message = "Start-HomePacs-Activity",
            homeViewVM = null,
            interpretationViewVM = null
        )
    }

    fun getHomeViewVM(): HomeViewVM? {
        return viewState.homeViewVM
    }
    fun printValueModel() {
        viewState.homeViewVM?.let {
            Log.w(TAG, "homeViewVM: ${it.studyPreview}")
        }
    }

    fun copyHomeViewVM(homeViewModel: HomeViewVM) {
        viewState = viewState.copy(homeViewVM = homeViewModel)
    }
    fun copyInterpretationViewVM(interpretationViewVM: InterpretationViewVM) {
        Log.w(TAG, "copyInterpretationViewVM")
        viewState = viewState.copy(interpretationViewVM = interpretationViewVM)
    }

    fun reduce(reducer: HomePacsReducer) {
        val result = reducer.reduce()
        reduce(result)
    }

    fun reduce(result: HomePacsObject) {
        result.viewState?.let { viewState = it }
        result.viewEffect?.let { viewEffect = it }
    }

    fun getInterpretationViewVM(): InterpretationViewVM? {
        return viewState.interpretationViewVM
    }

    fun checkIsNewStudy(): Boolean {
        viewState.homeViewVM?.let { homeViewVM ->
            viewState.interpretationViewVM?.let {interpretationViewVM ->
                return homeViewVM.studyID != interpretationViewVM.studyID
            }
        }
        return true
    }

}