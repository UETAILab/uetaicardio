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

class InterpretationEmptyMVI {

    companion object {
        // For Singleton instantiation
        @Volatile
        private var instance: InterpretationEmptyMVI? = null

        fun getInstance() =
            instance ?: synchronized(this) {
                instance
                    ?: InterpretationEmptyMVI()
                        .also { instance = it }
            }

        const val TAG = "InterpretationEmptyMVI"

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

    }

    fun process(interpretationActVM: InterpretationActVM, interpretationViewEvent: InterpretationViewEvent) {

        when (interpretationViewEvent) {

        }
    }

}