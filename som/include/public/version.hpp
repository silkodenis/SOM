/* Copyright 2018 Denis Silko. All rights reserved.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http:www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef version_h
#define version_h

#define SOM_VERSION_MAJOR    1
#define SOM_VERSION_MINOR    0
#define SOM_VERSION_REVISION 0
#define SOM_VERSION_STATUS   ""

#define SOMAUX_STR_EXP(__A)  #__A
#define SOMAUX_STR(__A)      SOMAUX_STR_EXP(__A)

#define SOMAUX_STRW_EXP(__A)  L ## #__A
#define SOMAUX_STRW(__A)      SOMAUX_STRW_EXP(__A)

#define SOM_VERSION          SOMAUX_STR(SOM_VERSION_MAJOR) "." SOMAUX_STR(SOM_VERSION_MINOR) "." SOMAUX_STR(SOM_VERSION_REVISION) SOM_VERSION_STATUS

#endif /* version_h */
