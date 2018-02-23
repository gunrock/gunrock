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
      sh '''cd build
            cmake ..
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

    def msg = "${buildStatus}: `${env.JOB_NAME}` #${env.BUILD_NUMBER}:\n${env.BUILD_URL}"

    slackSend(color: color, message: msg)
}

pipeline {
  agent any
  triggers {
    pollSCM '@hourly'
  }
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
              ctest -VV'''
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
        // Implement: junit 'build/reports/**/*.xml'  
        notifySlack('SUCCESS')
      }
      failure {
        notifySlack('FAILURE')
      }
   }
}
