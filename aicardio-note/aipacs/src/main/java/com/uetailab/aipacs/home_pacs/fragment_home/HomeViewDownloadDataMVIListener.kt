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
import com.uetailab.aipacs.home_pacs.HomePacsAPI
import com.uetailab.aipacs.home_pacs.LCE
import kotlinx.coroutines.launch

class HomeViewDownloadDataMVIListener(val downloadListener: HomePacsAPI.ProgressDownloadListener) {


    companion object {
        const val TAG = "DownloadDataHomeViewMVIListener"
    }
    private val homeViewRepository = HomeViewRepository.getInstance()

    private fun renderViewEffect(homeViewFragment: HomeViewFragment, viewEffect: HomeViewEffect) {

    }

    private fun renderViewState(homeViewFragment: HomeViewFragment, viewState: HomeViewState) {

    }

    fun process(homeViewVM: HomeViewVM, homeViewEvent: HomeViewEvent) {

        when (homeViewEvent) {
            is HomeViewEvent.FetchPreviewStudy -> {
                homeViewVM.viewStates().value?.let {
                    homeViewVM.reduce(FetchPreviewStudyReducerAsync(homeViewVM, it, homeViewEvent))
                }
            }

            is HomeViewEvent.FetchFileMP4 -> {
                homeViewVM.viewStates().value?.let {
                    homeViewVM.reduce(FetchPreviewStudyReducerAsync(homeViewVM, it, homeViewEvent))
                }
            }
        }
    }

    inner class FetchPreviewStudyReducerAsync(viewModel: HomeViewVM, viewState: HomeViewState, val viewEvent: HomeViewEvent) : HomeViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): HomeViewObject {
            viewModel.viewModelScope.launch {
                when (viewEvent) {
                    is HomeViewEvent.FetchPreviewStudy -> {
                        when (val resultLaunch = homeViewRepository.downloadAndSaveStudyPreview(listener=downloadListener, studyID=viewEvent.studyID, studyInstanceUID=viewEvent.studyInstanceUID)) {
                            is LCE.Result -> {
                                if (resultLaunch.error) {
                                    // cause error
                                    viewModel.viewStates().value?.let {
                                        viewModel.reduce(HomeViewObject(
                                            viewState = it.copy(status = HomeViewStatus.FetchedErrorData(viewEvent)),
                                            viewEffect = HomeViewEffect.ShowToast("Error Fetch Preview Study")))
                                    }
                                } else {
                                    Log.w(TAG, "Study preview: ${resultLaunch.data}")
                                    viewModel.viewStates().value?.let {
                                        viewModel.reduce(HomeViewObject(
                                            viewState = it.copy(studyPreview = resultLaunch.data, status = HomeViewStatus.FetchedData(viewEvent)),
                                            viewEffect = HomeViewEffect.ShowToast("Fetched Success Preview Study")
                                        ))
                                    }
                                }
                            }

                        }

                    }

                    is HomeViewEvent.FetchFileMP4 -> {
                        when (val resultLaunch = homeViewRepository.downloadAndExtractMP4File(downloadListener,
                            studyID = viewEvent.studyID, studyInstanceUID = viewEvent.studyInstanceUID, relativePath = viewEvent.relativePath
                            )) {
                            is LCE.Result -> {
                                if (resultLaunch.error) {
                                    // cause error
                                    viewModel.viewStates().value?.let {
                                        viewModel.reduce(HomeViewObject(
                                            viewState = it.copy(status = HomeViewStatus.FetchedErrorData(viewEvent), bitmaps = emptyList(), relativePath = viewEvent.relativePath),
                                            viewEffect = HomeViewEffect.ShowToast("Error Fetch File MP4 ${viewEvent.relativePath}")))
                                    }
                                } else {
                                    Log.w(TAG, "Relative path mp4 #frame: ${resultLaunch.data.size}")
                                    viewModel.viewStates().value?.let {
                                        viewModel.reduce(HomeViewObject(
                                            viewState = it.copy(bitmaps = resultLaunch.data, relativePath = viewEvent.relativePath, status = HomeViewStatus.FetchedData(viewEvent))
//                                            viewEffect = HomeViewEffect.ShowToast("Fetched Success File MP4 ${viewEvent.relativePath}")
                                        ))
                                    }
                                }
                            }

                        }
                    }
                }


            }
            return HomeViewObject(viewState=viewState.copy(status = HomeViewStatus.OnFetchingData(viewEvent)))
        }

    }
}
