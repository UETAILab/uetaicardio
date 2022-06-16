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

package com.ailab.aicardiotrainer.nav_interpretation

import android.Manifest
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.GravityCompat
import androidx.navigation.findNavController
import androidx.navigation.ui.setupWithNavController
import com.ailab.aicardiotrainer.R
import com.ailab.aicardiotrainer.nav_interpretation.ui.annotation.AnnotationFragment
import com.ailab.aicardiotrainer.nav_interpretation.ui.annotation.checkAndRequestPermissions
import com.google.android.material.navigation.NavigationView
import kotlinx.android.synthetic.main.activity_navigation_interpretation.*
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader


class NavigationInterpretationActivity : AppCompatActivity(), NavigationView.OnNavigationItemSelectedListener {

    companion object {
        const val TAG = "NavigationInterpretationActivity"
        const val MY_PERMISSIONS_REQUEST_CODE = 1

    }
    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_navigation_interpretation)


        checkAndRequestPermissions(
            this, arrayOf(
                Manifest.permission.INTERNET,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
        )



//        bottom_navigation.setOnNavigationItemSelectedListener(object : BottomNavigationView.OnNavigationItemSelectedListener {
//
//            override fun onNavigationItemSelected(item: MenuItem): Boolean {
//
//                when (item.getItemId()) {
//                    R.id.navigation_notifications -> toast("Select Notification")
//                    R.id.navigation_dashboard -> toast("Select Dashboard")
//                    R.id.navigation_home -> toast("Select Home")
//                }
//
//                return true
//            }
//        })

        // Passing each menu ID as a set of Ids because each
        // menu should be considered as top level destinations.

//        val navController = findNavController(R.id.nav_host_fragment)

//        val appBarConfiguration = AppBarConfiguration(
//            setOf(
//                R.id.navigation_home, R.id.navigation_dashboard, R.id.navigation_notifications
//            )
//        )

//        setupActionBarWithNavController(navController, appBarConfiguration)

        bottom_navigation.setupWithNavController(findNavController(R.id.nav_host_fragment))
//        bottom_nav_menu.
//        drawer_nav_view.setNavigationItemSelectedListener(this)

//        if (savedInstanceState == null) {
//            supportFragmentManager.beginTransaction()
//                .replace(R.id.nav_host_fragment, AnnotationFragment.getInstance())
//                .commitNow()
//        }


    }

//    override fun onCreateOptionsMenu(menu: Menu): Boolean {
//        // Inflate the menu; this adds items to the action bar if it is present.
//        menuInflater.inflate(R.menu.drawer_navigation, menu)
//        return true
//    }

//    override fun onSupportNavigateUp(): Boolean {
//        val navController = findNavController(R.id.nav_host_fragment)
//        return navController.navigateUp(appBarConfiguration) || super.onSupportNavigateUp()
//    }

    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                SUCCESS -> {
                    Log.i(AnnotationFragment.TAG, "OpenCV loaded successfully")
//                    mOpenCvCameraView.enableView()
//                    mOpenCvCameraView.setOnTouchListener(this@MainActivity)
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(AnnotationFragment.TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(AnnotationFragment.TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
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

    override fun onNavigationItemSelected(item: MenuItem): Boolean {
        // Handle navigation view item clicks here.
        when (item.itemId) {
            R.id.nav_gallery -> {
                Toast.makeText(this, "CLick Gallary Fragment Nav", Toast.LENGTH_SHORT).show()

            }
            R.id.nav_slideshow -> {
                Toast.makeText(this, "CLick Slideshow Fragment Nav", Toast.LENGTH_SHORT).show()

            }

            R.id.nav_camera -> {
                Toast.makeText(this, "CLick Camera Fragment Nav", Toast.LENGTH_SHORT).show()

            }
        }
        //close navigation drawer
        //close navigation drawer

//        drawer_layout.closeDrawer(GravityCompat.START)

        return true
    }

}