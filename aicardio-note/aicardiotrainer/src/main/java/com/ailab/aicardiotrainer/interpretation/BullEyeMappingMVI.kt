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

package com.ailab.aicardiotrainer.interpretation

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

import android.util.Log
import androidx.lifecycle.viewModelScope
import kotlinx.android.synthetic.main.activity_interpretation.*
import kotlinx.coroutines.launch
import java.util.*
import kotlin.collections.ArrayList
import kotlin.random.Random

class BullEyeMappingMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: BullEyeMappingMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: BullEyeMappingMVI()
                        .also { instance = it }
            }

        const val TAG = "BullEyeMappingMVI"

        fun process(interpretationActVM: InterpretationActVM, InterpretationViewEvent: InterpretationViewEvent) {
            getInstance().process(interpretationActVM, InterpretationViewEvent)
        }

        fun renderViewState(interpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {
            getInstance().renderViewState(interpretationActivity, viewState)
        }

        fun renderViewEffect(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect) {
            getInstance().renderViewEffect(interpretationActivity, viewEffect)
        }
    }


    private fun renderViewEffect(interpretationActivity: InterpretationActivity, viewEffect: InterpretationViewEffect) {

    }

    private fun renderViewState(interpretationActivity: InterpretationActivity, viewState: InterpretationViewState) {
        when (viewState.status) {
            InterpretationViewStatus.DoneGLSCalculating -> {
                interpretationActivity.showBullEyeMappingDialog(viewState.gls)
            }
        }
    }

    fun process(interpretationActVM: InterpretationActVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {
            InterpretationViewEvent.ShowBullEyeMapping -> {
                interpretationActVM.viewStates().value?.let {
                    interpretationActVM.reduce(GLSCalculatorReducer(
                        interpretationActVM, it, interpretationViewEvent
                    ))
                }
            }
        }
    }

    inner class GLSCalculatorReducer(viewModel: InterpretationActVM, viewState: InterpretationViewState, val viewEvent: InterpretationViewEvent)
        : InterpretationActReducer(viewModel, viewState, viewEvent){
        override fun reduce(): StateEffectObject {
            val result = StateEffectObject(viewState = viewState.copy(status = InterpretationViewStatus.OnGLSCalculating))

            viewModel.viewModelScope.launch {
                val gls = getGLS()
                viewModel.reduce(
                    StateEffectObject(
                    viewState = viewState.copy(gls = gls, status = InterpretationViewStatus.DoneGLSCalculating),
                    viewEffect = null)
                )
            }
            return result
        }

        fun getGLS(): List<Float> {
            val a = ArrayList<Float>()
            repeat(18) {
                val v = -20.0F + 40.0F * Random.nextFloat()
                a.add(v)
            }
            return a
        }
    }



}