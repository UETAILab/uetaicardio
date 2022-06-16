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

import aacmvi.AacMviActivity
import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.ArrayAdapter
import androidx.activity.viewModels
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.mainscreen.MainActivity
import com.ailab.aicardiotrainer.studyscreen.StudyActivity
import com.ailab.aicardiotrainer.toast
import kotlinx.android.synthetic.main.activity_search_case.*

class SearchCaseActivity : AacMviActivity<SearchCaseState, SearchCaseEffect, SearchCaseEvent, SearchCaseVM>() {

    companion object {
        const val TAG = "SearchCaseActivity"
        fun getListCase(): ArrayList<Int>{
            val cases = arrayListOf<Int>()
            repeat( 5000) {
                cases.add(it + 1) // add from [1, 2, 3, ..]
//                Log.w(TAG, "${it}")
            }
            return cases
        }

        fun createIntent(mainActivity: MainActivity): Intent? {
            val intent = Intent(mainActivity,
                SearchCaseActivity::class.java)
            return intent
        }

    }
    val cases = getListCase()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_search_case)

        checkAndRequestPermissions(this, arrayOf(
            Manifest.permission.INTERNET,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        ))

        val adapter = ArrayAdapter(this,
            android.R.layout.simple_list_item_1, cases)

        auto_complete_case_index.setAdapter(adapter)

        btn_submit_case.setOnClickListener {
            Log.w(TAG, "Case id get: ${auto_complete_case_index.text}")
            this.toast("Case id get: ${auto_complete_case_index.text}")
            auto_complete_case_index.text?.let {
                startActivity(StudyActivity.createIntent(this, it.toString().toInt()))
            }

        }

    }

    override val viewModel: SearchCaseVM  by viewModels()

    override fun renderViewState(viewState: SearchCaseState) {
//        TODO("Not yet implemented")

        when (viewState.status) {
            is SearchCaseStatus.Start -> {
                this.toast("Start search case")
            }
        }
    }

    override fun renderViewEffect(viewEffect: SearchCaseEffect) {
//        TODO("Not yet implemented")

    }

    fun checkAndRequestPermissions(activity: Activity, permissions: Array<String>, MY_PERMISSIONS_REQUEST_CODE: Int = MainActivity.MY_PERMISSIONS_REQUEST_CODE): Boolean {
        // Here, thisActivity is the current activity
        var granted = true
        permissions.forEach { permission ->
            if (ContextCompat.checkSelfPermission(
                    activity,
                    permission
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                granted = false
                Log.w(MainActivity.TAG, "checkAndRequestPermissions NOT granted $permission")
            }
            else {
                Log.w(MainActivity.TAG, "checkAndRequestPermissions granted $permission")
            }
        }

        return if (!granted) {
            // Permission is not granted
            // Should we show an explanation?
            ActivityCompat.requestPermissions(
                activity,
                permissions,
                MY_PERMISSIONS_REQUEST_CODE
            )
            false
        } else {
            true
        }
//        log.warning("Permission granted = $permissionGranted")
    }

}