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

package com.uetailab.aipacs.home_pacs.fragment_home

import androidx.lifecycle.viewModelScope
import com.uetailab.aipacs.home_pacs.HomePacsAPI
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class HomeViewMVIListener(val listener: HomePacsAPI.ProgressDownloadListener) {

    companion object {
        const val TAG = "HomeViewMVIListener"
    }


    private fun renderViewEffect(homeViewFragment: HomeViewFragment, viewEffect: HomeViewEffect) {

    }

    private fun renderViewState(homeViewFragment: HomeViewFragment, viewState: HomeViewState) {

    }

    fun process(homeViewVM: HomeViewVM, homeViewEvent: HomeViewEvent) {

        when (homeViewEvent) {

        }
    }

    inner class ReducerAsync(viewModel: HomeViewVM, viewState: HomeViewState, viewEvent: HomeViewEvent) : HomeViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): HomeViewObject {
            viewModel.viewModelScope.launch {
                val result = asyncRun()
                viewModel.viewStates().value?.let {
                    viewModel.reduce(HomeViewObject())
                }
            }
            return HomeViewObject()
        }

        suspend fun asyncRun() : Int = withContext(Dispatchers.IO) {
            repeat(100000000) {
                100*100
            }
            return@withContext 1234
        }
    }
}
