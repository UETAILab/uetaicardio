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

import android.util.Log
import androidx.lifecycle.viewModelScope
import com.uetailab.aipacs.home_pacs.LCE
import com.uetailab.aipacs.home_pacs.fragment_home.HomeViewRepository.Companion.KEY_STUDY_INSTANCE_UID
import kotlinx.coroutines.launch

class HomeViewFetchDataMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: HomeViewFetchDataMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: HomeViewFetchDataMVI()
                        .also { instance = it }
            }

        const val TAG = "FetchDataHomeViewMVI"

        fun process(homeViewVM: HomeViewVM, HomeViewEvent: HomeViewEvent) {
            getInstance().process(homeViewVM, HomeViewEvent)
        }

        fun renderViewState(homeViewFragment: HomeViewFragment, viewState: HomeViewState) {
            getInstance().renderViewState(homeViewFragment, viewState)
        }

        fun renderViewEffect(homeViewFragment: HomeViewFragment, viewEffect: HomeViewEffect) {
            getInstance().renderViewEffect(homeViewFragment, viewEffect)
        }
    }

    private fun renderViewEffect(homeViewFragment: HomeViewFragment, viewEffect: HomeViewEffect) {

    }

    private fun renderViewState(homeViewFragment: HomeViewFragment, viewState: HomeViewState) {

    }
    private val homeViewRepository = HomeViewRepository.getInstance()
    fun process(homeViewVM: HomeViewVM, homeViewEvent: HomeViewEvent) {

        when (homeViewEvent) {
            is HomeViewEvent.FetchStudies -> {
                homeViewVM.viewStates().value?.let {
                    homeViewVM.reduce(FetchReducerAsync(homeViewVM, it, homeViewEvent))
                }
            }
            is HomeViewEvent.FetchStudy -> {
                homeViewVM.viewStates().value?.let {
                    homeViewVM.reduce(FetchReducerAsync(homeViewVM, it, homeViewEvent))
                }
            }
        }
    }

    inner class FetchReducerAsync(viewModel: HomeViewVM, viewState: HomeViewState, val viewEvent: HomeViewEvent) : HomeViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): HomeViewObject {

            viewModel.viewModelScope.launch {
                when (viewEvent) {

                    is HomeViewEvent.FetchStudies -> {
                        when (val resultLaunch = homeViewRepository.getListStudies(typeData = viewEvent.typeData)) {
                            is LCE.Result -> {
                                Log.w(TAG, "resultLaunch.error: ${resultLaunch.error}")
                                if (resultLaunch.error) {
                                    // cause error
                                    viewModel.viewStates().value?.let {
                                        viewModel.reduce(HomeViewObject(
                                            viewState = it.copy(status = HomeViewStatus.FetchedErrorData(viewEvent)),
                                            viewEffect = HomeViewEffect.ShowToast("Fetch Studies Error")
                                        ))
                                    }
                                } else {
                                    viewModel.viewStates().value?.let {
                                        viewModel.reduce(HomeViewObject(
                                            viewState = it.copy(studies = resultLaunch.data, status = HomeViewStatus.FetchedData(viewEvent)),
                                            viewEffect = HomeViewEffect.ShowToast("Done Fetched Studies")
                                        ))
                                    }
                                }

                            }
                        }
                    }

                    is HomeViewEvent.FetchStudy -> {
                        val studyID6d = "%06d".format(viewEvent.studyID)
                        when (val resultLaunch = homeViewRepository.getInformationStudy(studyID = viewEvent.studyID)) {
                            is LCE.Result -> {
                                if (resultLaunch.error) {
                                    // cause error
                                    viewModel.viewStates().value?.let {
                                        viewModel.reduce(HomeViewObject(
                                            viewState = it.copy(status = HomeViewStatus.FetchedErrorData(viewEvent)),
                                            viewEffect = HomeViewEffect.ShowToast("Fetch Study ${viewEvent.studyID} Error")
                                        ))
                                    }
                                } else {
//                                    Log.w(TAG, "${KEY_STUDY_INSTANCE_UID} : ${resultLaunch.data.first.getString(KEY_STUDY_INSTANCE_UID)}")
//                                    Log.w(TAG, "MetaData: ${resultLaunch.data}")
                                    // resultLaunch.data.first: metadata and view chamber of each case
                                    // resultLaunch.data.second: annotation of the case
                                    viewModel.viewStates().value?.let {
                                        viewModel.reduce(HomeViewObject(
                                            viewState = it.copy(studyID = studyID6d, bitmaps = emptyList(),
                                                studyInstanceUID = resultLaunch.data.first.getString(KEY_STUDY_INSTANCE_UID),
                                                studyMetaData = resultLaunch.data.first,
                                                studyInterpretation = resultLaunch.data.second,
                                                status = HomeViewStatus.FetchedData(viewEvent)),
                                                viewEffect = HomeViewEffect.ShowToast("Done Fetched Study")
                                        ))
                                    }
                                }

                            }
                        }
                    }


                }


            }
            return HomeViewObject(viewState = viewState.copy(status = HomeViewStatus.OnFetchingData(viewEvent)))
        }

    }
}
