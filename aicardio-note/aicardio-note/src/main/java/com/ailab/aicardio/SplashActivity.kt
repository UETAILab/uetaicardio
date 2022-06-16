/*
 * Copyright 2020 ET-AILAB
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

package com.ailab.aicardio

import android.os.Bundle
import android.os.Handler
import androidx.appcompat.app.AppCompatActivity
import com.ailab.aicardio.mainscreen.MainActivity
import com.ailab.aicardio.repository.FolderRepository

import java.util.logging.Logger

class SplashActivity : AppCompatActivity() {

    @Suppress("unused")
    val log: Logger = Logger.getLogger(SplashActivity::class.java.name)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

        /**
         * delay 2 seconds for show working_heart.gif image
         * Calling LoginActivity
         */
        Handler().postDelayed({
            log.warning("Calling HomeActivity")
            val intent = MainActivity.createIntent(this, folder = FolderRepository.DEFAULT_FOLDER_DOWNLOAD) //Intent(this, NavigationActivity::class.java)
            startActivity(intent)
            finish()
        }, 2000)
    }
}
