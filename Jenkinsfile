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

pipeline {
  agent any
  stages {
    stage('Init') {
      steps {
        init_git()
        slackSend(token: 'Nq0oASH7cBXxDKOiO3oW5NpA', teamDomain: 'https://gunrock.slack.com', baseUrl: 'https://gunrock.slack.com/services/hooks/jenkins-ci/', channel: '#builds', message: 'Pipeline started build and tests (gunrock:master).')
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
      parallel {
        stage('Deploy') {
          steps {
            echo 'Branch: Master.'
            echo 'Pipleline finished.'
          }
        }
        stage('Slack') {
          steps {
            slackSend(token: 'Nq0oASH7cBXxDKOiO3oW5NpA', teamDomain: 'https://gunrock.slack.com', baseUrl: 'https://gunrock.slack.com/services/hooks/jenkins-ci/', channel: '#builds', message: 'Pipeline finished build and tests (gunrock:master).')
          }
        }
      }
    }
  }
  post { 
      always { 
          cleanWs()
      }
   }
}
