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
import com.uetailab.aipacs.home_pacs.HomePacsAPI
import com.uetailab.aipacs.home_pacs.LCE
import kotlinx.coroutines.launch

class InterpretationViewDownloadDataMVIListener(val downloadListener: HomePacsAPI.ProgressDownloadListener) {


    companion object {
        const val TAG = "DownloadDataInterpretationViewMVIListener"
    }
    private val interpretationViewRepository = InterpretationViewRepository.getInstance()

    private fun renderViewEffect(interpretationViewFragment: InterpretationViewFragment, viewEffect: InterpretationViewEffect) {

    }

    private fun renderViewState(interpretationViewFragment: InterpretationViewFragment, viewState: InterpretationViewState) {

    }

    fun process(interpretationViewVM: InterpretationViewVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {

            is InterpretationViewEvent.FetchFileMP4 -> {
                interpretationViewVM.viewStates().value?.let {
                    interpretationViewVM.reduce(FetchPreviewStudyReducerAsync(interpretationViewVM, it, interpretationViewEvent))
                }
            }
        }
    }

    inner class FetchPreviewStudyReducerAsync(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent) : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {

            // click into current file then do nothing (don't load mp4 file)
            if (viewEvent is InterpretationViewEvent.FetchFileMP4 && viewModel.relativePath == viewEvent.relativePath)
                return InterpretationViewObject(viewEffect = InterpretationViewEffect.ShowToast(message = "Clicked Current file Playing"))

            viewModel.viewModelScope.launch {
                when (viewEvent) {

                    is InterpretationViewEvent.FetchFileMP4 -> {


                        when (val resultLaunch = interpretationViewRepository.downloadAndExtractMP4File(downloadListener,
                            studyID = viewEvent.studyID, studyInstanceUID = viewEvent.studyInstanceUID, relativePath = viewEvent.relativePath)) {
                            is LCE.Result -> {
                                if (resultLaunch.error) {
                                    // cause error
                                    viewModel.viewStates().value?.let {
                                        viewModel.reduce(InterpretationViewObject(
                                            viewState = it.copy(status = InterpretationViewStatus.FetchedErrorData(viewEvent)),
                                            viewEffect = InterpretationViewEffect.ShowToast("Error Fetch File MP4 ${viewEvent.relativePath}")))
                                    }
                                } else {
                                    Log.w(TAG, "Relative path mp4 #frame: ${resultLaunch.data.size}")
                                    // co duoc video thi set trang thai playing = True
                                    viewModel.setIsPlaying(true)
//                                    if (resultLaunch.data.size > 0) {
//                                        if (resultLaunch.data.size == 1) viewModel.setIsPlaying(false)
//                                        else viewModel.setIsPlaying(true)
//                                    }
                                    viewModel.viewStates().value?.let {
                                        // check push data in to studyInterpretation
                                        val fileDicom = viewEvent.relativePath
                                        val oldValue = if (it.studyInterpretation.has(fileDicom)) DicomInterpretation(it.studyInterpretation.getJSONObject(fileDicom)) else DicomInterpretation(resultLaunch.data.size)

                                        var dicomInterpretation = if (oldValue.getJSONArray(DicomInterpretation.KEY_ANNOTATION).length() == resultLaunch.data.size) oldValue else DicomInterpretation(resultLaunch.data.size)
                                        var machineInterpretation = DicomInterpretation(resultLaunch.data.size)

                                        val dicomMetadata = it.getDicomMetaData(viewEvent.relativePath)
                                        Log.w(TAG, "dicomMetadata: ${dicomMetadata}")
                                        viewModel.reduce(InterpretationViewObject(
                                            viewState = it.copy(bitmaps = resultLaunch.data,
                                                dicomInterpretation = dicomInterpretation,
                                                machineInterpretation = machineInterpretation,

                                                studyInterpretation = it.studyInterpretation.put(viewEvent.relativePath, dicomInterpretation),

                                                relativePath = viewEvent.relativePath,
                                                dicomMetaData = dicomMetadata,

                                                status = InterpretationViewStatus.FetchedData(viewEvent))
//                                            viewEffect = InterpretationViewEffect.ShowToast("Fetched Success File MP4 ${viewEvent.relativePath}")
                                        ))
                                    }
                                }
                            }

                        }
                    }
                }


            }
            return InterpretationViewObject(viewState=viewState.copy(status = InterpretationViewStatus.OnFetchingData(viewEvent)))
        }

    }
}
