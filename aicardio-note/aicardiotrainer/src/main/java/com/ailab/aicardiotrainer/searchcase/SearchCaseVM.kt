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

package com.ailab.aicardiotrainer.searchcase

import android.app.Application
import aacmvi.AacMviViewModel
import com.ailab.aicardiotrainer.studyscreen.StudyActReducer
import com.ailab.aicardiotrainer.studyscreen.StudyStateEffectObject

class SearchCaseVM(application: Application) : AacMviViewModel<SearchCaseState, SearchCaseEffect, SearchCaseEvent>(application) {
    companion object {
        const val TAG = "SearchCaseVM"
    }
    init {
        viewState = SearchCaseState(
            status = SearchCaseStatus.Start,
            id_case = -1
        )
    }



    fun reduce(reducer: SearchCaseActReducer) {
        val result = reducer.reduce()
        reduce(result)
    }



    fun reduce(result: SearchCaseStateEffectObject) {
        result.viewState?.let { viewState = it }
        result.viewEffect?.let { viewEffect = it }
    }
}