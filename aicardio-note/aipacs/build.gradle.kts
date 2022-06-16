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
import org.jetbrains.kotlin.config.KotlinCompilerVersion
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmOptions

plugins {
    id("com.android.application")
    id("kotlin-android")
    id("kotlin-android-extensions")

}
repositories {
    mavenCentral()

    maven {
        setUrl("https://jitpack.io")
    }
}

android {
    compileSdkVersion(29)

    defaultConfig {
        applicationId = "com.example.aipacs"
        minSdkVersion(28)
        targetSdkVersion(29)
        versionCode=1
        versionName= "1.0"
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")

        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }

    (kotlinOptions as KotlinJvmOptions).apply {
        jvmTarget = JavaVersion.VERSION_1_8.toString()
    }
    sourceSets {
        getByName("main") {
            res {
                srcDirs("src/main/res", "src/main/res/layout/home_pacs", "src/main/res/layouts/home_pacs", "src/main/res/layouts/interpretation")
            }
        }
    }

}

dependencies {
    // Minimum implementation for empty project
    implementation(kotlin("stdlib-jdk8", KotlinCompilerVersion.VERSION))
    implementation("androidx.appcompat:appcompat:1.2.0")
    implementation("com.google.android.material:material:1.3.0")

    //ViewModel and LiveData (for import in file by ViewModels)
    implementation("androidx.activity:activity-ktx:1.1.0")
    implementation("androidx.navigation:navigation-fragment-ktx:2.1.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.2.0")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.2.0")


    //Webservices
    implementation("com.squareup.retrofit2:retrofit:2.6.2")
    implementation("com.squareup.okhttp3:logging-interceptor:4.0.1")
    implementation("androidx.constraintlayout:constraintlayout:2.0.4")
    implementation("androidx.navigation:navigation-fragment:2.3.3")
    implementation("androidx.navigation:navigation-ui:2.3.3")
    implementation("androidx.navigation:navigation-ui-ktx:2.3.3")

    //Image Loading
    implementation("io.coil-kt:coil:0.8.0")

    // Android bar chart
    implementation("com.github.PhilJay:MPAndroidChart:v3.1.0")
    // com.github.PhilJay:MPAndroidChart:v3.0.0-beta1
}