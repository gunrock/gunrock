// Jenkins pipeline
// See documents at https://jenkins.io/doc/book/pipeline/jenkinsfile/

// initialize source codes
def init_git() {
  checkout scm
  retry(5) {
    timeout(time: 2, unit: 'MINUTES') {
      sh 'git submodule update --init'
    }
  }
}

// build gunrock using cmake
def cmake_build() {
  checkout scm
  retry(5) {
    timeout(time: 20, unit: 'MINUTES') {
      sh 'mkdir -p build'
      // MGPU not fully supported in v1.0.0
      sh '''cd build
            cmake -DGUNROCK_CODE_COVERAGE=ON -DGUNROCK_GOOGLE_TESTS=ON .. //-DGUNROCK_MGPU_TESTS=ON ..
            make -j16'''
    }
  }
}

// notify slack of build status and progress
def notifySlack(String buildStatus = 'STARTED') {
    // Build status of null means success.
    buildStatus = buildStatus ?: 'SUCCESS'

    def color

    if (buildStatus == 'STARTED') {
        color = '#D4DADF'
    } else if (buildStatus == 'SUCCESS') {
        color = '#BDFFC3'
    } else if (buildStatus == 'UNSTABLE') {
        color = '#FFFE89'
    } else {
        color = '#FF9FA1'
    }

    def msg = "${buildStatus}: `${env.JOB_NAME}` #${env.BUILD_NUMBER}:\n${env.RUN_DISPLAY_URL}"

    slackSend(color: color, message: msg)
}

pipeline {
  agent any
  stages {
    stage('Init') {
      steps {
        notifySlack('STARTED')
        init_git()
      }
    }

    stage('Build') {
      steps {
        cmake_build()
      }
    }
    
    stage('Regression Tests') {
      steps {
        sh '''cd build
              cd examples
              ctest -VV'''
      }
    }
    
    stage('Code Coverage') {
      steps {
        sh '''#!/bin/bash
              cd build
              CODECOV_TOKEN="d0690e81-c2ed-42d0-8a63-da351c3ae619"
              bash <(curl -s https://codecov.io/bash) -t ${CODECOV_TOKEN} || echo "Error: Codecov did not collect coverage reports"'''
      }
    }
    
    stage('Deploy') {
      steps {
        echo 'Branch: Master.'
        echo 'Pipleline finished.'
      }
    }
  }
  
  post {
    always {
      cleanWs() 
    }
    success {
      notifySlack('SUCCESS')
    }
    failure {
      notifySlack('FAILURE')
    }
  }
}
