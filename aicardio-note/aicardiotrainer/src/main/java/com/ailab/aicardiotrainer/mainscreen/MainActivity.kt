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

package com.ailab.aicardiotrainer.mainscreen

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.viewModels
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.ailab.aicardiotrainer.*
import com.ailab.aicardiotrainer.interfaces.OnStudyClicked
import com.ailab.aicardiotrainer.repositories.SkillRepository
import com.ailab.aicardiotrainer.repositories.StudyItem
import com.ailab.aicardiotrainer.searchcase.SearchCaseActivity
import com.ailab.aicardiotrainer.studyscreen.StudyActivity
import com.rohitss.aacmvi.AacMviActivity
import kotlinx.android.synthetic.main.activity_main.*

class MainActivity
    :  AacMviActivity<TrainerState, TrainerViewEffect, TrainerViewEvent, TrainerActVM>(),
      OnStudyClicked {

    companion object {
        const val TAG = "MainActivity"
        const val MY_PERMISSIONS_REQUEST_CODE = 1
    }

    private var rvSkillsAdapter = SkillRVAdapter(this)
    private val skillList = SkillRepository.getInstance().getSkillList()

    override fun onCreate(savedInstanceState: Bundle?) {
        System.loadLibrary("imebra_lib")

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        var granted = checkAndRequestPermissions(this, arrayOf(
            Manifest.permission.INTERNET,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        ))

//        Log.w(TAG, "${granted}")
//        if (granted) {
//            startActivity(SearchCaseActivity.createIntent(this))
//        }

        rv_skills.apply {
            adapter = rvSkillsAdapter
        }

        rvSkillsAdapter.submitList(skillList)

    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        when (requestCode) {
            MY_PERMISSIONS_REQUEST_CODE -> {
                permissions.forEach { permission ->
                    Log.w(TAG, "onRequestPermissionsResult $permission")
                }
            }
        }
    }

    override fun onStudyClicked(skillName: String, study: StudyItem) {
        Log.w(TAG, "skill ${skillName} study ${study.name}")
//        viewModel.process(TrainerViewEvent.DownloadStudy(skillName, study))
        startActivity(StudyActivity.createIntent(this, skillName, study))
    }

    override val viewModel: TrainerActVM by viewModels()

    override fun renderViewState(viewState: TrainerState) {
    }

    override fun renderViewEffect(viewEffect: TrainerViewEffect) {
    }

    fun checkAndRequestPermissions(activity: Activity, permissions: Array<String>, MY_PERMISSIONS_REQUEST_CODE: Int = Companion.MY_PERMISSIONS_REQUEST_CODE): Boolean {
        // Here, thisActivity is the current activity
        var granted = true
        permissions.forEach { permission ->
            if (ContextCompat.checkSelfPermission(
                    activity,
                    permission
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                granted = false
                Log.w(TAG, "checkAndRequestPermissions NOT granted $permission")
            }
            else {
                Log.w(TAG, "checkAndRequestPermissions granted $permission")
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
