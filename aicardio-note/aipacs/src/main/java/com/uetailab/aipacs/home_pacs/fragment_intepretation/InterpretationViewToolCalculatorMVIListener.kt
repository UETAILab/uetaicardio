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

class InterpretationViewToolCalculatorMVIListener(val downloadListener: HomePacsAPI.ProgressDownloadListener) {


    companion object {
        const val TAG = "InterpretationViewToolCalculatorMVIListener"
    }
    private val interpretationViewRepository = InterpretationViewRepository.getInstance()

    private fun renderViewEffect(interpretationViewFragment: InterpretationViewFragment, viewEffect: InterpretationViewEffect) {

    }

    private fun renderViewState(interpretationViewFragment: InterpretationViewFragment, viewState: InterpretationViewState) {

    }

    fun process(interpretationViewVM: InterpretationViewVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {

            is InterpretationViewEvent.OnAutoCalculateEFGLS -> {
                interpretationViewVM.viewStates().value?.let {
                    interpretationViewVM.reduce(FetchAutoEFGLSStudyReducerAsync(interpretationViewVM, it, interpretationViewEvent))
                }
            }
        }
    }

    inner class FetchAutoEFGLSStudyReducerAsync(viewModel: InterpretationViewVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent) : InterpretationViewReducer(viewModel, viewState, viewEvent) {

        override fun reduce(): InterpretationViewObject {

            val relativePath = viewModel.relativePath
            val studyInstanceUID = viewModel.studyInstanceUID
            val metaDataDicom = viewModel.dicomMetadata
            Log.w(TAG, "Start FetchAutoEFGLSStudyReducerAsync: ${relativePath} ${studyInstanceUID} ${metaDataDicom}  ${viewModel.studyID} ")

            if (relativePath != null && studyInstanceUID != null) {

                viewModel.viewModelScope.launch {
                    when (viewEvent) {

                        is InterpretationViewEvent.OnAutoCalculateEFGLS -> {
                            Log.w(TAG, "InterpretationViewEvent.FetchFileMP4 ${relativePath} ${studyInstanceUID} ")

//                            when (val resultLaunch = interpretationViewRepository.uploadStudyMP4File(
//                                listener = downloadListener,
//                                relativePath = relativePath,
//                                studyID = studyID,
//                                metadata = metaDataDicom
//                            ))

                            when (val resultLaunch = interpretationViewRepository.getAutoEFForRelativePath(
                                listener = downloadListener,
                                studyInstanceUID = studyInstanceUID,
                                relativePath = relativePath

                            )){
                                is LCE.Result -> {
                                    if (resultLaunch.error) {
                                        // cause error
                                        viewModel.viewStates().value?.let {
                                            viewModel.reduce(
                                                InterpretationViewObject(
                                                    viewState = it.copy(status = InterpretationViewStatus.FetchedErrorData(viewEvent)),
                                                    viewEffect = InterpretationViewEffect.ShowToast("Error Fetch AUTO EF GLS file: ${relativePath}")
                                                )
                                            )
                                        }
                                    } else {
//                                        Log.w(TAG, "FetchAutoEFGLSStudyReducerAsync : ${resultLaunch.data}")
                                        Log.w(TAG, "FetchAutoEFGLSStudyReducerAsync keys: ${resultLaunch.data.getJSONObject("data").keys()}")
                                        viewModel.viewStates().value?.let {
                                            viewModel.reduce(
                                                InterpretationViewObject(
                                                    viewState = it.copy(status = InterpretationViewStatus.FetchedData(viewEvent),
                                                        glsValue = resultLaunch.data.getJSONObject("data").getDouble("gls").toFloat(),
//                                                        efValue = resultLaunch.data.getJSONObject("data").getDouble("ef").toFloat(),
//                                                        dicomInterpretation =  DicomInterpretation(resultLaunch.data.getJSONObject("data")),
                                                        machineInterpretation = DicomInterpretation(resultLaunch.data.getJSONObject("data"))

                                                    ),
                                                    viewEffect = InterpretationViewEffect.ShowToast("Done Fetch AUTO EF GLS file: ${relativePath}")
                                                )
                                            )
                                        }

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
