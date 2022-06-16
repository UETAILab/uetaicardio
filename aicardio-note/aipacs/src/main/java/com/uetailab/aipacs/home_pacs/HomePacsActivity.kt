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

package com.uetailab.aipacs.home_pacs

import aacmvi.AacMviActivity
import android.Manifest
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import androidx.activity.viewModels
import androidx.navigation.NavController
import androidx.navigation.NavDestination
import androidx.navigation.findNavController
import androidx.navigation.ui.setupWithNavController
import com.uetailab.aipacs.R
import com.uetailab.aipacs.home_pacs.fragment_home.HomeViewFragment
import com.uetailab.aipacs.home_pacs.fragment_home.HomeViewVM
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewFragment
import com.uetailab.aipacs.home_pacs.fragment_intepretation.InterpretationViewVM
import com.uetailab.aipacs.home_pacs.fragment_intepretation.OnHomeViewDataPass
import kotlinx.android.synthetic.main.activity_home_pacs.*


class HomePacsActivity : AacMviActivity<HomePacsState, HomePacsEffect, HomePacsEvent, HomePacsVM>(),
    HomeViewFragment.OnHomViewVMPass,
    InterpretationViewFragment.OnInterpretationViewVMPass {
    // other link useful: https://github.com/moallemi/MultiNavHost/blob/master/app/src/main/java/me/moallemi/multinavhost/navigation/TabManager.kt
    companion object {
        const val TAG = "HomePacsActivity"
    }

    lateinit var homeViewDataPass: OnHomeViewDataPass

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home_pacs)

        checkAndRequestPermissions(this, arrayOf(
            Manifest.permission.INTERNET,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
        ))

        val navController = findNavController(R.id.nav_host_fragment)


        navController.addOnDestinationChangedListener { controller: NavController?, destination: NavDestination, arguments: Bundle? ->
            when (destination.id) {
                R.id.navigation_home -> {
//                    viewModel.printValueModel()
//                    Log.w(TAG, "R.id.navigation_home")
                }
                R.id.navigation_interpretation -> {
                    Log.w(TAG, "R.id.navigation_interpretation")
                }
                R.id.navigation_server -> {
//                    viewModel.printValueModel()
//                    Log.w(TAG, "R.id.navigation_server")
                }
            }

        }
        bottom_navigation.setupWithNavController(navController)

    }
    override val viewModel: HomePacsVM by viewModels()

    override fun renderViewState(viewState: HomePacsState) {
//        TODO("Not yet implemented")
    }

    override fun renderViewEffect(viewEffect: HomePacsEffect) {
//        TODO("Not yet implemented")
    }

    fun getHomeViewVM(): HomeViewVM? {
        return viewModel.getHomeViewVM()
    }

    fun getInterpretationViewVM(): InterpretationViewVM? {
        return viewModel.getInterpretationViewVM()
    }

    override fun onHomeViewVMPass(homeViewVM: HomeViewVM) {
        viewModel.copyHomeViewVM(homeViewVM)
    }

    override fun onInterpretationViewVMPass(interpretationViewVM: InterpretationViewVM) {
        viewModel.copyInterpretationViewVM(interpretationViewVM)
    }

    fun checkIsNewStudy(): Boolean {
        return viewModel.checkIsNewStudy()
    }

}
